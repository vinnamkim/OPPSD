import argparse
import os

import numpy as np
import pandas as pd
import torch
from captum.attr import (AttributionalCornerDetection, GuidedBackprop,
                         GuidedGradCam, IntegratedGradients,
                         LocalPatchGradientAttribution, NoiseTunnel, Saliency)
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision_xai_tools.datasets import get_dataset, get_subset_dataset
from vision_xai_tools.utils import get_model


def get_attr_method(model: nn.Module, attr_method_name: str):
    if attr_method_name == 'gradient':
        attr = Saliency(model)
        return lambda inputs, target: attr.attribute(inputs, target, abs=False).mean(1)
    elif attr_method_name == 'gradient-sq-sum':
        attr = Saliency(model)
        return lambda inputs, target: attr.attribute(inputs, target, abs=False).norm(p=2, dim=1)
    elif attr_method_name == 'ig':
        attr = IntegratedGradients(model)
        return lambda inputs, target: attr.attribute(
            inputs=inputs, target=target, internal_batch_size=64, n_steps=32).mean(1)
    elif 'local' in attr_method_name:
        splited = attr_method_name.split('-')
        kernel_size = int(splited[1])
        attr = LocalPatchGradientAttribution(model)
        return lambda inputs, target: attr.attribute(
            inputs, target, kernel_size=kernel_size).mean(1)
    elif 'pcd' in attr_method_name:
        splited = attr_method_name.split('-')
        method = splited[1]
        kernel_type = splited[2]
        kernel_size = int(splited[3])
        if kernel_type == 'gaussian':
            kernel_sigma = float(splited[4])
        else:
            kernel_sigma = None

        attr = AttributionalCornerDetection(model)
        return lambda inputs, target: attr.attribute(
            inputs=inputs, target=target,
            kernel_type=kernel_type, kernel_size=kernel_size, kernel_sigma=kernel_sigma,
            method=method, num_samples=1000,
            force_double_type=True).mean(1)
    elif 'vargrad' in attr_method_name:
        attr = NoiseTunnel(Saliency(model))
        return lambda inputs, target: attr.attribute(
            inputs=inputs, target=target,
            abs=False, nt_type='vargrad', n_samples=16, stdevs=0.225).mean(1)
    elif 'smoothgrad_sq' in attr_method_name:
        attr = NoiseTunnel(Saliency(model))
        return lambda inputs, target: attr.attribute(
            inputs=inputs, target=target,
            abs=False, nt_type='smoothgrad_sq', n_samples=16, stdevs=0.225).mean(1)
    elif 'guided-grad-cam' == attr_method_name:
        layer = None

        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                print(n)
                layer = m

        attr = GuidedGradCam(model, layer)
        return lambda inputs, target: attr.attribute(inputs, target).mean(1)
    elif 'guided-backprop' == attr_method_name:
        attr = GuidedBackprop(model)
        return lambda inputs, target: attr.attribute(inputs, target).mean(1)
    elif 'fullgrad' == attr_method_name:
        from .fullgrad.saliency.fullgrad import FullGrad
        fullgrad = FullGrad(model)
        return lambda inputs, target: fullgrad.saliency(inputs, target).mean(1)
    else:
        raise Exception(f'Unknown method {attr_method_name}')


parser = argparse.ArgumentParser()
parser.add_argument('--model-name', '-n', type=str, default='resnet50')
parser.add_argument('--method', '-m', type=str, required=True)
parser.add_argument('--dataset', '-d', type=str, required=True)
parser.add_argument('--split', '-s', type=str, required=True)
parser.add_argument('--randomize', '-r', type=str, required=False)
parser.add_argument('--batch-size', '-b', type=int, default=32)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--subset', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_name = args.dataset
    split = args.split
    attr_method_name = args.method

    if attr_method_name == 'vargrad' or attr_method_name == 'smoothgrad_sq':
        batch_size = 2
    elif 'pcd-min' in attr_method_name:
        batch_size = 2
    elif 'guided-grad-cam' == attr_method_name:
        batch_size = 32
    elif attr_method_name == 'fullgrad':
        batch_size = 8
    elif attr_method_name == 'ig':
        batch_size = 1

    print(
        f'Start {model_name} {dataset_name} {split} {attr_method_name} {batch_size}')

    if not args.subset:
        dataset = get_dataset(dataset_name, split)
    else:
        dataset = get_subset_dataset(dataset_name, split)

    print('Dataset transform : ', dataset.transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    model = get_model(dataset_name, model_name, args.randomize)

    attr_method = get_attr_method(model, attr_method_name)

    results = []

    i = 0

    for batch in tqdm(dataloader):
        inputs, target, fnames = batch[0].cuda(), batch[1].cuda(), batch[2]
        inputs.requires_grad = True
        attrs = attr_method(inputs, target).detach().cpu().to(
            dtype=torch.float)
        results += [(fname, attr.flatten().numpy())
                    for attr, fname in zip(attrs, fnames)]

        if args.debug is True and i > 3:
            break
        # print(attrs.shape)
        # print(results[0][1].shape)
        i += 1

    # df = pd.DataFrame(results)
    # df.columns = ['fname', 'attr']

    if args.randomize is not None:
        attr_method_name += "-" + args.randomize

    dirpath = os.path.join('attrs',
                           dataset_name, model_name, attr_method_name, split)

    if os.path.exists(dirpath):
        import shutil
        print(f'Remove {dirpath}.')
        shutil.rmtree(dirpath)

    os.makedirs(dirpath, exist_ok=True)

    print(f'Save to {dirpath}.')
    for result in tqdm(results):
        fname, attr = result
        fpath = os.path.join(dirpath, os.path.splitext(fname)[0])

        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))

        np.save(fpath, attr)
