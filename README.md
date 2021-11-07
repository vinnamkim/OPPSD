# One-Step Pixel-Level Perturbation-Based Saliency Detector

## If you just want to use our algorithm

### How to setup
1. Prepare Python 3.8 environment with PyTorch `torch==1.7.1+cu110`.
2. Git clone this repo.
```shell
$ git clone https://github.com/vinnamkim/OPPSD.git
$ cd OPPSD
```
3. Git submodule update with initialization.
```shell
$ git submodule update --init
```
4. Install dependency.
```shell
$ pip install -e ./deps/captum
```

### How to use
 - Our algorithm is actually implemented on top of [Captum](https://github.com/pytorch/captum). It is implemented [here](https://github.com/vinnamkim/captum/blob/master/captum/attr/_core/attributional_corner_detection.py).
 - We include our algorithm as a Git submodule in `./deps/captum`.
 - Please refer to the following [example](./example.py).
```python
import torch
from captum.attr import AttributionalCornerDetection
from torchvision.models.resnet import resnet50

model = resnet50().eval()

attr = AttributionalCornerDetection(model)

# Arbitary inputs
inputs = torch.randn(1, 3, 224, 224).requires_grad_(True)
target = torch.LongTensor([0])

kernel_type: str = 'window'  # ['window', 'gaussian']
kernel_size: int = 7
kernel_sigma: float = 0.7
method: str = 'noble'  # ['noble', 'fro', 'exact-min', 'min']

# Obtain a saliency map
saliency_map = attr.attribute(
    inputs=inputs, target=target,
    kernel_type=kernel_type, kernel_size=kernel_size, kernel_sigma=kernel_sigma,
    method=method, num_samples=1000,
    force_double_type=True).mean(1)

print(saliency_map.shape)
```

## If you want to reproduce our paper experiments
## How to setup
1. Prepare Python 3.8 environment with PyTorch `torch==1.7.1+cu110`.
2. Git clone this repo.
```shell
$ git clone https://github.com/vinnamkim/OPPSD.git
$ cd OPPSD
```
3. Git submodule update with initialization.
```shell
$ git submodule update --init
```
4. Install dependency.
```shell
$ pip install -r requirements.txt
```
5. Setup datasets. Please read [./dataset/README.md](./dataset/README.md) for more details.
