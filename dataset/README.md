# ILSVRC 2012 dataset
 - Extract ILSVRC 2012 validation dataset to `./dataset/imagenet2012/val`
```
📦dataset
 ┣ 📂imagenet2012
 ┃ ┗ 📂val
 ┃ ┃ ┣ 📂n01440764
 ┃ ┃ ┃ ┣ 📜ILSVRC2012_val_00000293.JPEG
```

# CIFAR100 dataset
 - You can download CIFAR100 dataset from `torchvision` code. It will automatically extract the dataset to `./dataset/cifar-100-python`.
```
📦dataset
 ┣ 📂cifar-100-python
 ┃ ┣ 📜file.txt~
 ┃ ┣ 📜meta
 ┃ ┣ 📜test
 ┃ ┗ 📜train
```

# Food101 dataset
 - Link: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
 - Download [food-101.tar.gz](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) and extract it to this directory.
 - Then run `python food101.py` to split it into train and valid directories.
```
📦dataset
 ┣ 📂food-101
 ┃ ┣ 📂train
 ┃ ┃ ┣ 📂apple_pie
 ┃ ┃ ┃ ┣ 📜1005649.jpg
```

# Bird250 dataset
 - Link (Version 35): https://www.kaggle.com/gpiosenka/100-bird-species/version/35
 - DownLoad `train.zip` and `valid.zip` from the link. Then extract them to `./dataset/bird250/train` and `./dataset/bird250/valid`.
 - 🔴IMPORTANT, we used Version 35 dataset that includes 250 species of birds.
```
📦dataset
 ┣ 📂bird250
 ┃ ┣ 📂train
 ┃ ┃ ┣ 📂AFRICAN CROWNED CRANE
 ┃ ┃ ┃ ┣ 📜001.jpg
```
