import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision_xai_tools.datasets import get_dataset, get_perturbed_dataset
from vision_xai_tools.utils import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', '-n', type=str, default='resnet50')
parser.add_argument('--dataset', '-d', type=str, required=True)
parser.add_argument('--method', '-m', type=str, required=True)
parser.add_argument('--percentage', '-p', type=float, required=True)
parser.add_argument('--batch-size', '-b', type=int, default=32)
parser.add_argument('--im', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_name = args.dataset
    split = 'val'
    attr_method_name = args.method
    percentage = args.percentage
    remove_important = args.im

    print(f'Start {model_name} {dataset_name} {split} {attr_method_name} {percentage} remove_important : {remove_important}')

    if attr_method_name == 'origin':
        dataset = get_dataset(dataset_name, split)
    else:
        dataset = get_perturbed_dataset(
            dataset_name=args.dataset,
            attrs_path=f'./attrs/{args.dataset}/{model_name}/{args.method}',
            percentage=args.percentage,
            remove_important=remove_important,
            split='valid')

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=12)
    model = get_model(dataset_name, model_name, None)

    results = []

    i = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, target, fnames = batch[0].cuda(), batch[1].cuda(), batch[2]
            logits = model(inputs)

            results += [
                (fname, t, logit)
                for logit, t, fname
                in zip(logits.detach().cpu().numpy(), target.detach().cpu().numpy(), fnames)]

            # if i > 3:
            #     break
            i += 1

    df = pd.DataFrame(results)
    df.columns = ['fname', 'target', 'logit']

    dirpath = os.path.join(
        'logits',
        dataset_name,
        model_name,
        'imp' if remove_important else 'unimp')
    os.makedirs(dirpath, exist_ok=True)

    fname = '_'.join([attr_method_name, split, f'{percentage:.3f}'])
    fpath = os.path.join(dirpath, fname + '.parquet')

    if os.path.exists(fpath):
        print(f'{fpath} already exists.')
        os.remove(fpath)

    df.to_parquet(fpath)
