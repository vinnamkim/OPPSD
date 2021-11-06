import argparse
from models.resnet import resnet50
import os

import numpy as np
import pandas as pd
import torch
from captum.attr import (AttributionalCornerDetection, GuidedBackprop,
                         GuidedGradCam, IntegratedGradients,
                         LocalPatchGradientAttribution, NoiseTunnel, Saliency)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vision_xai_tools.datasets import get_dataset
from vision_xai_tools.utils import get_model
from time import sleep, time


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
parser.add_argument('--randomize', '-r', type=str, required=False)
parser.add_argument('--batch-size', '-b', type=int, default=32)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    batch_size = args.batch_size
    model_name = args.model_name
    attr_method_name = args.method

    print(
        f'Start {model_name} {attr_method_name} {batch_size}')

    dataset = TensorDataset(
        torch.randn(
            [128 * 8, 3, 224, 224]),
        torch.zeros([128 * 8], dtype=torch.long)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    model = get_model('food101', model_name)

    attr_method = get_attr_method(model, attr_method_name)

    results = []

    i = 0

    start = time()
    for batch in tqdm(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.cuda(), labels.cuda()
        # print(inputs.shape)
        inputs.requires_grad = True
        attrs = attr_method(inputs, labels).detach().cpu().to(
            dtype=torch.float)

        if args.debug is True and i > 3:
            break
        # print(attrs.shape)
        # print(results[0][1].shape)
        i += 1

    end = time()

    try:
        df = pd.read_csv('time.csv', index_col=0)
    except Exception as e:
        df = pd.DataFrame(
            columns=['model_name', 'method_name', 'batch_size', 'elapsed_time', 'num_samples'])

    df = df.append({'model_name': model_name, 'method_name': attr_method_name, 'batch_size': batch_size,
                    'elapsed_time': end-start, 'num_samples': len(dataset)},
                   ignore_index=True)

    df.to_csv('time.csv')
