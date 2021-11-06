import argparse
import os

import torch

from vision_xai_tools.datasets import (Bird250XRAIDataset, Food101XRAIDataset,
                                       ImageNetXRAIDataset)
from vision_xai_tools.utils import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', '-n', type=str, default='resnet50')
parser.add_argument('--method', '-m', type=str, required=True)
parser.add_argument('--dataset', '-d', type=str, required=True)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


def get_dataset(dataset_name: str, method_name: str):
    attr_path = f'attrs/{dataset_name}/resnet50/{method_name}'
    if dataset_name == 'imagenet':
        return ImageNetXRAIDataset(
            attr_path, 0.0, False, True)
    elif dataset_name == 'bird250':
        return Bird250XRAIDataset(
            attr_path, 0.0, False, True)
    elif dataset_name == 'food101':
        return Food101XRAIDataset(
            attr_path, 0.0, False, True)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    dataset = get_dataset(args.dataset, args.method)

    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=32)

    import pandas as pd
    from torch.nn.functional import softmax

    model = get_model(args.dataset, 'resnet50')

    dfs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [b.squeeze(0) for b in batch]
            features, targets, percentages = batch

            outputs = model(features.cuda())

            probs = softmax(outputs, 1).gather(
                1, targets.cuda().unsqueeze(-1)).squeeze(-1).cpu()

            target_probs = probs / probs[-1]

            logits, preds = outputs.topk(1)

            accs = preds.cpu().squeeze(-1) == targets

            df = pd.DataFrame(
                (percentages.numpy(), target_probs.numpy(), accs.numpy(),)).T

            df.columns = ['percentages', 'logits', 'accs']

            dfs += [df]

            if args.debug is True and len(dfs) > 4:
                break

    dfs = pd.concat(dfs, axis=0)

    dirpath = os.path.join('xrai', args.dataset)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    fpath = os.path.join(dirpath, f'{args.method}.parquet')

    if os.path.exists(fpath):
        os.remove(fpath)

    dfs.to_parquet(fpath)
