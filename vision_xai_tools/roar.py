import argparse
from catalyst.engines.apex import APEXEngine
from catalyst.engines.torch import DeviceEngine

import torch
from catalyst.dl import SupervisedRunner
from catalyst.callbacks import (AccuracyCallback, CheckpointCallback, CriterionCallback,
                                OptimizerCallback, SchedulerCallback)

from vision_xai_tools.datasets import get_perturbed_dataset
from vision_xai_tools.scheduler import get_multi_step_lr_with_warmup
from vision_xai_tools.utils import get_model

#from utils.cifar100_models import resnet34, resnet50, vgg16_bn


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--dataset', '-d', type=str, required=True)
parser.add_argument('--percentage', '-p', type=float, required=True)
parser.add_argument('--path', type=str, default='roar')
args = parser.parse_args()


def get_train_params(dataset: str, model_name: str):
    if model_name == 'resnet50':
        if dataset == 'cifar100':
            batch_size = 128
            num_epochs = 200
            milestones = [60, 120, 160]
        elif dataset == 'food101':
            batch_size = 64
            num_epochs = 120
            milestones = [30, 60, 90]
        elif dataset == 'bird225':
            batch_size = 64
            num_epochs = 120
            milestones = [30, 60, 90]
        elif dataset == 'bird250':
            batch_size = 64
            num_epochs = 120
            milestones = [30, 60, 90]
        else:
            raise NotImplementedError()

        lr = 0.1
        gamma = 0.2
        weight_decay = 5e-4
        accumulation_steps = 1

        engine = DeviceEngine('cuda')

        return batch_size, num_epochs, milestones, lr, gamma, weight_decay, accumulation_steps, engine
    else:
        batch_size = 64
        num_epochs = 10
        lr = 6e-4
        gamma = 0.5
        weight_decay = 1e-4
        milestones = [30, 60, 90]
        accumulation_steps = 512 // batch_size

        engine = APEXEngine(apex_kwargs={'opt_level': 'O1'})

        return batch_size, num_epochs, milestones, lr, gamma, weight_decay, accumulation_steps, engine


def datasets_fn():
    train_dataset = get_perturbed_dataset(
        dataset_name=args.dataset,
        attrs_path=f'./attrs/{args.dataset}/resnet50/{args.method}',
        percentage=args.percentage,
        split='train')
    valid_dataset = get_perturbed_dataset(
        dataset_name=args.dataset,
        attrs_path=f'./attrs/{args.dataset}/resnet50/{args.method}',
        percentage=args.percentage,
        split='valid')
    return {
        'train': {
            'dataset': train_dataset, 'shuffle': True},
        'valid': {
            'dataset': valid_dataset, 'shuffle': False}}


if __name__ == "__main__":
    batch_size, num_epochs, milestones, lr, gamma, weight_decay, accumulation_steps, engine = get_train_params(
        args.dataset, args.model)

    from apex import amp

    model = get_model(args.dataset, args.model, load_ckpt=False)

    if args.model == 'vit':
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=lr, weight_decay=weight_decay)

        model, optimizer = amp.initialize(model, optimizer)
    else:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=lr, momentum=0.9, weight_decay=weight_decay)

    train_dataset = datasets_fn()['train']['dataset']
    num_iters_per_epoch = (len(train_dataset) // batch_size) + 1

    scheduler = get_multi_step_lr_with_warmup(
        optimizer,
        num_warmup_epochs=1,
        epoch_milestones=milestones,
        num_iters_per_epochs=num_iters_per_epoch,
        gamma=gamma)

    criterion = torch.nn.CrossEntropyLoss()

    runner = SupervisedRunner()

    logdir = f'./{args.path}/{args.dataset}/{args.model}/{args.method}/{args.percentage:.2f}'

    runner.train(
        model=model,
        engine=engine,
        optimizer=optimizer,
        scheduler={'scheduler': scheduler},
        loaders={
            'batch_size': batch_size,
            'num_workers': 8,
            'datasets_fn': datasets_fn,
        },
        callbacks={
            'accuracy': AccuracyCallback(input_key='logits', target_key='targets'),
            'criterion': CriterionCallback(
                input_key="logits", target_key="targets", metric_key="loss"
            ),
            'checkpoint': CheckpointCallback(loader_key='valid', metric_key='accuracy01', minimize=False, logdir=logdir),
            'optimizer': OptimizerCallback(metric_key='loss', accumulation_steps=accumulation_steps),
            'scheduler': SchedulerCallback('scheduler', 'batch'),
        },
        valid_metric='accuracy01',
        minimize_valid_metric=False,
        criterion=criterion,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=False,
        check=False,
        fp16=False,
    )
