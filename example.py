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
