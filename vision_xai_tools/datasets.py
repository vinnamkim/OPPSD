import math
import os

import numpy as np
import torch
from numpy.core.defchararray import index
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.cifar import CIFAR100
from torchvision.datasets.imagenet import ImageNet

from vision_xai_tools.xrai import (_get_segments_felzenszwalb, _xrai,
                                   get_webp_length)


class ImageNetDataset(ImageNet):
    def __init__(self,
                 split: str = 'valid',
                 normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                 **kwargs):
        if split == 'valid':
            split = 'val'

        imagenet_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        self.normalize = normalize

        super().__init__(root='./dataset/imagenet2012/',
                         split=split, transform=imagenet_transforms)

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        returns = super().__getitem__(index)
        fname = os.path.split(path)[-1]
        return returns + (fname,)


class ImageNetDatasetSubset(ImageNetDataset):
    def __init__(self,
                 split: str = 'valid'):
        super().__init__(split=split)

        np.random.seed(5151)

        self.indices = np.random.randint(0, super().__len__(), [500])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        real_index = self.indices[index]
        return super().__getitem__(real_index)


class PerturbedDatasetInterface:
    def __init__(self,
                 attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool,
                 split: str = 'valid',
                 is_bird: bool = False, **kwargs):
        print('Init PerturbedDatasetInterface')
        super().__init__(split=split, **kwargs)

        self.attrs_path = attrs_path
        self.attrs = {}

        root_path = os.path.join(attrs_path, split)

        if is_bird is True:
            for root, dirs, files in os.walk(root_path):
                if len(files) > 0:
                    rel_path = os.path.relpath(root, root_path)
                    for file in files:
                        fpath = os.path.join(rel_path, file)
                        self.attrs[os.path.splitext(fpath)[0]] = os.path.join(
                            attrs_path, split, fpath)

        else:
            for fpath in os.listdir(os.path.join(attrs_path, split)):
                fname = os.path.splitext(fpath)[0]
                self.attrs[fname] = os.path.join(attrs_path, split, fpath)

        self.percentage = percentage
        self.remove_important = remove_important
        self.abs = abs

    def get_fname(self, image_path: str):
        fpath = os.path.split(image_path)[-1]
        return os.path.splitext(fpath)[0]

    def get_attr(self, image_path: str):
        fname = self.get_fname(image_path)
        attr_fpath = self.attrs[fname]
        fattr = torch.FloatTensor(np.load(attr_fpath))
        return fattr.abs() if self.abs else fattr

    def get_mask(self, fattr: torch.Tensor):
        k = max(1, int(len(fattr) * self.percentage))
        indices = fattr.argsort(descending=self.remove_important)[:k]

        width = height = math.isqrt(fattr.shape[0])

        mask = torch.zeros_like(
            fattr, dtype=torch.bool).scatter_(0, indices, True).reshape(1, width, height)

        return mask

    def fill_mask(self, input_image: torch.Tensor, mask: torch.Tensor):
        channels = input_image.shape[0]
        num_masks = mask.sum()
        mask = mask.repeat(channels, 1, 1)
        mean_input = input_image.mean([1, 2])

        input_image[mask] = mean_input.repeat(
            num_masks).reshape(num_masks, -1).T.flatten()

        return input_image

    def get_perturbed_image(self, input_image: torch.Tensor, image_path: str):
        fattr = self.get_attr(image_path)
        mask = self.get_mask(fattr)
        return self.fill_mask(input_image, mask)


class ImageNetPerturbedDataset(PerturbedDatasetInterface, ImageNetDataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):
        super().__init__(
            attrs_path=attrs_path,
            percentage=percentage,
            remove_important=remove_important,
            abs=abs,
            split=split)

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target, image_fname = super().__getitem__(index)
        perturbed_image = self.get_perturbed_image(input_image, image_path)
        fname = self.get_fname(image_path)

        return perturbed_image, target, fname


class XRAIDatasetInterface(PerturbedDatasetInterface):
    def __init__(self,
                 attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool,
                 split: str = 'valid',
                 is_bird: bool = False, **kwargs) -> None:
        super().__init__(attrs_path, percentage, remove_important, abs, split, is_bird)

    def get_xrai_inputs(self, input_image, target, image_path):
        float_image = input_image * self.std + self.mean

        uint8_image = np.uint8(float_image.permute(1, 2, 0).numpy() * 255.)
        mean = uint8_image.mean(0).mean(0)
        bg_image = mean.astype(np.uint8).reshape(
            1, 1, 3).repeat(224, 0).repeat(224, 1)

        full_length = get_webp_length(uint8_image)

        segs = _get_segments_felzenszwalb(uint8_image)
        attr = self.get_attr(image_path).reshape(224, 224)
        fname = self.get_fname(image_path)

        output_attr, masks_trace = _xrai(attr, segs, integer_segments=False)

        idx = 5

        perturbed_images = []
        percentages = []

        for mask in masks_trace:
            bg_image[mask] = uint8_image[mask]

            length = get_webp_length(bg_image, debug=0)

            percentage = length / full_length

            if int(percentage * 100) >= idx:
                idx += 5

                perturbed_images += [
                    (torch.FloatTensor(bg_image).permute(2, 0, 1) / 255. - self.mean) / self.std]

                percentages += [percentage]

                if percentage > 1.:
                    break

        perturbed_images = torch.stack(perturbed_images)

        perturbed_images[-1] = input_image

        percentages = torch.FloatTensor(percentages)

        percentages[-1] = 1.

        return perturbed_images, torch.LongTensor([target] * len(perturbed_images)), percentages


class Food101Dataset(ImageFolder):
    NORMALIZE = transforms.Normalize(
        [0.54930437, 0.44500041, 0.34350203],
        [0.272926, 0.27589517, 0.27998645])
    AUG_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        NORMALIZE
    ])
    NORM_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE
    ])

    def __init__(self, split: str = 'train', augmentation: bool = False):
        transform = Food101Dataset.AUG_TRANSFORM if augmentation else Food101Dataset.NORM_TRANSFORM

        if split == 'train':
            dirname = 'train'
        elif split == 'valid':
            dirname = 'test'
        else:
            raise Exception('Unknown split : {split}')

        super().__init__(
            root=os.path.join('dataset', 'food-101', dirname),
            transform=transform
        )

    def get_fname(self, image_path: str):
        fpath = os.path.split(image_path)[-1]
        return os.path.splitext(fpath)[0]

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        input_image, target = super().__getitem__(index)

        fname = self.get_fname(path)

        return input_image, target, fname


class Food101PerturbedDataset(PerturbedDatasetInterface, Food101Dataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):
        super().__init__(attrs_path=attrs_path,
                         percentage=percentage,
                         remove_important=remove_important,
                         split=split,
                         abs=abs,
                         augmentation=False)
        print('Init Food101PerturbedDataset')

    def get_fname(self, image_path: str):
        return Food101Dataset.get_fname(self, image_path)

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target, image_fname = super().__getitem__(index)
        perturbed_image = self.get_perturbed_image(input_image, image_path)
        return perturbed_image, target


class Bird225Dataset(ImageFolder):
    NORMALIZE = transforms.Normalize(
        [0.4723, 0.4706, 0.3999],
        [0.2011, 0.1978, 0.2033])

    AUG_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    NORM_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE
    ])

    def __init__(self, split: str = 'train', augmentation: bool = False, **kwargs):
        print('Init Bird225Dataset')
        transform = Bird225Dataset.AUG_TRANSFORM if augmentation else Bird225Dataset.NORM_TRANSFORM

        super().__init__(
            root=os.path.join('dataset', 'bird225', split),
            transform=transform
        )

    def _get_statistics(self):
        dataset = ImageFolder(
            root=os.path.join('dataset', 'bird225', 'train'),
            transform=transforms.ToTensor())

        means = []
        stds = []

        from tqdm import tqdm
        for img, label in tqdm(dataset):
            means += [img.mean([1, 2])]
            stds += [img.std([1, 2])]

        print('Mean', torch.stack(means).mean(0))
        print('Std', torch.stack(stds).mean(0))

    def get_fname(self, image_path: str):
        fpath = '/'.join(image_path.split(os.sep)[-3:])
        return os.path.splitext(fpath)[0]

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target = super().__getitem__(index)
        fname = self.get_fname(image_path)

        return input_image, target, fname


class Bird225PerturbedDataset(PerturbedDatasetInterface, Bird225Dataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):

        if split != 'train':
            split = 'valid'

        super().__init__(attrs_path=attrs_path,
                         percentage=percentage,
                         remove_important=remove_important,
                         split=split,
                         abs=abs,
                         augmentation=False,
                         is_bird=True)
        print('Init Bird225PerturbedDataset')

    def get_fname(self, image_path: str):
        return Bird225Dataset.get_fname(self, image_path)

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target, image_fname = super().__getitem__(index)
        perturbed_image = self.get_perturbed_image(input_image, image_path)
        return perturbed_image, target


class Bird250Dataset(ImageFolder):
    NORMALIZE = transforms.Normalize(
        [0.4723, 0.4706, 0.3999],
        [0.2011, 0.1978, 0.2033])

    AUG_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    NORM_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE
    ])

    def __init__(self, split: str = 'train', augmentation: bool = False, **kwargs):
        print('Init Bird250Dataset')
        transform = Bird250Dataset.AUG_TRANSFORM if augmentation else Bird250Dataset.NORM_TRANSFORM

        super().__init__(
            root=os.path.join('dataset', 'bird250', split),
            transform=transform
        )

    def _get_statistics(self):
        dataset = ImageFolder(
            root=os.path.join('dataset', 'bird250', 'train'),
            transform=transforms.ToTensor())

        means = []
        stds = []

        from tqdm import tqdm
        for img, label in tqdm(dataset):
            means += [img.mean([1, 2])]
            stds += [img.std([1, 2])]

        print('Mean', torch.stack(means).mean(0))
        print('Std', torch.stack(stds).mean(0))

    def get_fname(self, image_path: str):
        fpath = '/'.join(image_path.split(os.sep)[-3:])
        return os.path.splitext(fpath)[0]

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target = super().__getitem__(index)
        fname = self.get_fname(image_path)

        return input_image, target, fname


class Bird250PerturbedDataset(PerturbedDatasetInterface, Bird250Dataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):

        if split != 'train':
            split = 'valid'

        super().__init__(attrs_path=attrs_path,
                         percentage=percentage,
                         remove_important=remove_important,
                         split=split,
                         abs=abs,
                         augmentation=False,
                         is_bird=True)
        print('Init Bird250PerturbedDataset')

    def get_fname(self, image_path: str):
        return Bird250Dataset.get_fname(self, image_path)

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target, image_fname = super().__getitem__(index)
        perturbed_image = self.get_perturbed_image(input_image, image_path)
        return perturbed_image, target


class CIFAR100Dataset(CIFAR100):
    NORMALIZE = transforms.Normalize(
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404), True)

    AUG_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        NORMALIZE
    ])

    NORM_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE
    ])

    def __init__(self, split: str = 'valid', augmentation: bool = False):
        transform = CIFAR100Dataset.AUG_TRANSFORM if augmentation else CIFAR100Dataset.NORM_TRANSFORM

        if split == 'train':
            super().__init__('dataset', transform=transform, train=True, download=True)
        elif split == 'valid':
            super().__init__('dataset', transform=transform, train=False, download=True)
        else:
            raise Exception()

    def __getitem__(self, index: int):
        return super().__getitem__(index) + (str(index), )


class CIFAR100PerturbedDataset(PerturbedDatasetInterface, CIFAR100Dataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):
        super().__init__(attrs_path=attrs_path,
                         percentage=percentage,
                         remove_important=remove_important,
                         abs=abs,
                         split=split,
                         augmentation=False)
        print('Init CIFAR100PerturbedDataset')

    def get_fname(self, image_path: str):
        return image_path

    def __getitem__(self, index: int):
        image_path = str(index)
        input_image, target, image_fname = super().__getitem__(index)
        perturbed_image = self.get_perturbed_image(input_image, image_path)
        return perturbed_image, target


class ImageNetXRAIDataset(XRAIDatasetInterface, ImageNetDataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):
        super().__init__(
            attrs_path=attrs_path,
            percentage=percentage,
            remove_important=remove_important,
            abs=abs,
            split=split)

        self.mean = torch.FloatTensor(
            self.normalize.mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.FloatTensor(
            self.normalize.std).unsqueeze(1).unsqueeze(1)
        print('Init ImageNetXRAIDataset')

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target, image_fname = super().__getitem__(index)

        return self.get_xrai_inputs(input_image, target, image_path)


class Bird250XRAIDataset(XRAIDatasetInterface, Bird250Dataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):
        super().__init__(
            attrs_path=attrs_path,
            percentage=percentage,
            remove_important=remove_important,
            abs=abs,
            split=split,
            is_bird=True)

        self.normalize = self.NORMALIZE
        self.mean = torch.FloatTensor(
            self.normalize.mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.FloatTensor(
            self.normalize.std).unsqueeze(1).unsqueeze(1)
        print('Init Bird250XRAIDataset')

    def get_fname(self, image_path: str):
        return Bird250Dataset.get_fname(self, image_path)

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target, image_fname = super().__getitem__(index)

        return self.get_xrai_inputs(input_image, target, image_path)


class Food101XRAIDataset(XRAIDatasetInterface, Food101Dataset):
    def __init__(self, attrs_path: str, percentage: float,
                 remove_important: bool, abs: bool, split: str = 'valid'):
        super().__init__(
            attrs_path=attrs_path,
            percentage=percentage,
            remove_important=remove_important,
            abs=abs,
            split=split)

        self.normalize = self.NORMALIZE
        self.mean = torch.FloatTensor(
            self.normalize.mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.FloatTensor(
            self.normalize.std).unsqueeze(1).unsqueeze(1)
        print('Init Food101XRAIDataset')

    def __getitem__(self, index: int):
        image_path, _ = self.samples[index]
        input_image, target, image_fname = super().__getitem__(index)

        return self.get_xrai_inputs(input_image, target, image_path)


def get_dataset(dataset_name: str, split: str, augmentation: bool = False) -> VisionDataset:
    if dataset_name == 'imagenet':
        return ImageNetDataset(split=split)
    elif dataset_name == 'cifar100':
        return CIFAR100Dataset(split=split)
    elif dataset_name == 'food101':
        return Food101Dataset(split=split, augmentation=augmentation)
    elif dataset_name == 'bird225':
        if split == 'train':
            return Bird225Dataset(split, augmentation=augmentation)
        elif split == 'valid':
            dataset = ConcatDataset([
                Bird225Dataset('valid', augmentation=augmentation),
                Bird225Dataset('test', augmentation=augmentation)])

            dataset.transform = dataset.datasets[0].transform
            return dataset
    elif dataset_name == 'bird250':
        if split == 'train':
            return Bird250Dataset(split, augmentation=augmentation)
        elif split == 'valid':
            dataset = ConcatDataset([
                Bird250Dataset('valid', augmentation=augmentation),
                Bird250Dataset('test', augmentation=augmentation)])

            dataset.transform = dataset.datasets[0].transform
            return dataset
    else:
        raise Exception(f'Uknown dataset_name : {dataset_name}')


def get_subset_dataset(dataset_name: str, split: str, augmentation: bool = False) -> VisionDataset:
    dataset = get_dataset(dataset_name, split, augmentation)

    class SubsetDataset(Dataset):
        def __init__(self, dataset: VisionDataset):
            super().__init__()
            self.dataset = dataset

            np.random.seed(5151)

            self.indices = np.random.randint(0, len(self.dataset), [500])
            self.transform = dataset.transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index: int):
            real_index = self.indices[index]
            return self.dataset.__getitem__(real_index)

    return SubsetDataset(dataset)


def get_perturbed_dataset(
        dataset_name: str,
        attrs_path: str,
        percentage: float,
        remove_important: bool = True,
        abs: bool = True,
        split: str = 'valid') -> VisionDataset:
    if dataset_name == 'imagenet':
        return ImageNetPerturbedDataset(
            attrs_path, percentage, remove_important, abs, split)
    elif dataset_name == 'cifar100':
        return CIFAR100PerturbedDataset(
            attrs_path, percentage, remove_important, abs, split)
    elif dataset_name == 'food101':
        return Food101PerturbedDataset(
            attrs_path, percentage, remove_important, abs, split)
    elif dataset_name == 'bird225':
        if split == 'train':
            return Bird225PerturbedDataset(
                attrs_path, percentage, remove_important, abs, split)
        elif split == 'valid':
            return ConcatDataset([
                Bird225PerturbedDataset(
                    attrs_path, percentage, remove_important, abs, 'valid'),
                Bird225PerturbedDataset(
                    attrs_path, percentage, remove_important, abs, 'test')])
    elif dataset_name == 'bird250':
        if split == 'train':
            return Bird250PerturbedDataset(
                attrs_path, percentage, remove_important, abs, split)
        elif split == 'valid':
            return ConcatDataset([
                Bird250PerturbedDataset(
                    attrs_path, percentage, remove_important, abs, 'valid'),
                Bird250PerturbedDataset(
                    attrs_path, percentage, remove_important, abs, 'test')])
    else:
        raise Exception(f'Uknown dataset_name : {dataset_name}')


def get_imagenet_perturbed_dataset(
        attr_method_name: str,
        model_name: str = 'resnet50',
        percentage: float = 0.1,
        remove_important: bool = False,
        abs: bool = True,
        split: str = 'valid'):

    attrs_path = os.path.join(
        'attrs', 'imagenet', model_name, attr_method_name)

    dataset = ImageNetPerturbedDataset(
        attrs_path=attrs_path, percentage=percentage,
        remove_important=remove_important, split=split, abs=abs)

    return dataset


if __name__ == "__main__":
    # print(Bird225Dataset('train').classes)
    # from torch.utils.data import ConcatDataset
    # datasets = ConcatDataset([Bird225Dataset('valid'), Bird225Dataset('test')])
    # print(datasets.classes)

    # dataset_name = 'cifar100'
    # dataset = get_perturbed_dataset(
    #     dataset_name=dataset_name,
    #     attrs_path=f'attrs/{dataset_name}/resnet50/ig',
    #     percentage=0.0,
    #     remove_important=False,
    #     abs=True,
    #     split='train'
    # )

    # dataset = ImageNetPerturbedDataset(
    #     attrs_path='attrs/imagenet/resnet50/gradient/val',
    #     percentage=0.1,
    #     remove_important=False)

    # from torchvision.utils import save_image
    # img, target, fname = dataset[5]
    # print(fname, target)
    # save_image(img / torch.FloatTensor(CIFAR100Dataset.NORMALIZE.std).reshape(3, 1, 1) +
    #            torch.FloatTensor(CIFAR100Dataset.NORMALIZE.mean).reshape(3, 1, 1), 'test.png')

    # datasets = ConcatDataset([Bird225Dataset('valid'), Bird225Dataset('test')])

    # dataset = get_imagenet_perturbed_dataset('pcd-noble-gaussian-9-2.5')

    # test = dataset[0]

    # print(test)

    # from torch.utils.data import ConcatDataset

    # dataset = Bird225Dataset('valid')
    # print(dataset[0][-1])

    # dataset = get_perturbed_dataset(
    #     'bird225', 'attrs/bird225/resnet50/gradient', 0.1, split='train')
    # print(dataset[0])

    dataset = Bird250XRAIDataset(
        'attrs/bird250/resnet50/gradient', 0.0, False, True)

    dataset[0]

    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=4)

    import pandas as pd
    from torch.nn.functional import softmax
    from torchvision.models.resnet import resnet50

    model = resnet50(pretrained=True).eval()

    dfs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [b.squeeze(0) for b in batch]
            features, targets, percentages = batch

            outputs = model(features)

            probs = softmax(outputs, 1).gather(
                1, targets.unsqueeze(-1)).squeeze(-1)

            target_probs = probs / probs[-1]

            logits, preds = outputs.topk(1)

            accs = preds.squeeze(-1) == targets

            df = pd.DataFrame(
                (percentages.numpy(), target_probs.numpy(), accs.numpy(),)).T

            df.columns = ['percentages', 'logits', 'accs']

            dfs += [df]

            if len(dfs) > 4:
                break

    dfs = pd.concat(dfs, axis=0)

    dfs.to_parquet('test.parquet')

    pass
