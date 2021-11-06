import os

import torch
from timm.models.vision_transformer import vit_base_patch16_224


def get_large_img_model(model_name: str, num_classes: int):
    from torchvision.models import resnet34, resnet50, vgg16, vgg16_bn

    if model_name == 'resnet50':
        return resnet50(pretrained=False, num_classes=num_classes).cuda().eval()
    elif model_name == 'resnet34':
        return resnet34(pretrained=False, num_classes=num_classes).cuda().eval()
    elif model_name == 'vgg16bn':
        return vgg16_bn(pretrained=False, num_classes=num_classes).cuda().eval()
    elif model_name == 'vgg16':
        return vgg16(pretrained=False, num_classes=num_classes).cuda().eval()
    elif model_name == 'vit':
        model = vit_base_patch16_224(pretrained=True, num_classes=num_classes)
        return model
    else:
        raise Exception(f'Unknown model_name {model_name}')


def get_small_img_model(model_name: str):
    from .pytorch_cifar100.models.resnet import resnet34, resnet50
    from .pytorch_cifar100.models.vgg import vgg16_bn

    if model_name == 'resnet50':
        model = resnet50().cuda().eval()
        return model
    elif model_name == 'resnet34':
        model = resnet34().cuda().eval()
        return model
    elif model_name == 'vgg16':
        model = vgg16_bn().cuda().eval()
        return model
    elif model_name == 'vit':
        class MyViT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.upsample = torch.nn.Upsample(
                    scale_factor=7, mode='bilinear')
                self.model = vit_base_patch16_224(
                    pretrained=True, num_classes=100)

            def forward(self, x):
                x = self.upsample(x)
                return self.model(x)

        model = MyViT()

        return model
    else:
        raise Exception(f'Unknown model_name {model_name}')


def get_model(dataset_name: str, model_name: str, randomize: str = None, load_ckpt: bool = True):
    num_classes = {
        'cifar100': 100,
        'food101': 101,
        'bird225': 225,
        'bird250': 250,
        'imagenet': 1000,
    }
    if dataset_name == 'cifar100':
        model = get_small_img_model(model_name)
    else:
        model = get_large_img_model(model_name, num_classes[dataset_name])

    if load_ckpt is True:
        if randomize is not None:
            ckpt_path = f'./checkpoints/{dataset_name}/{model_name}/{randomize}.pth'
        else:
            ckpt_path = f'./checkpoints/{dataset_name}/{model_name}/best.pth'

        if os.path.exists(ckpt_path):
            print(f'ckpt_path exists and load ckpt : {ckpt_path}')
            ckpt = torch.load(ckpt_path)['model_state_dict']
            model.load_state_dict(ckpt)
        else:
            raise Exception(f'ckpt_path : {ckpt_path}')
    else:
        print("Don't load ckpt")

    model = model.cuda().eval()

    return model
