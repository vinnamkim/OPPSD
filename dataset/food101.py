import os
from shutil import copy2
from tqdm import tqdm

def read_txt(path):
    with open(path, 'r') as fp:
        return fp.read().splitlines()

def create_dirs(path):
    os.makedirs(path, exist_ok=True)

    with open('food-101/meta/classes.txt', 'r') as fp:
        classes = fp.read().splitlines()
    
    for c in classes:
        os.makedirs(os.path.join(path, c), exist_ok=True)

# Create dirs
create_dirs('food-101/train')
create_dirs('food-101/valid')

splits = {
    'train': read_txt(f'food-101/meta/train.txt'),
    'valid': read_txt(f'food-101/meta/test.txt')
}

def copy(key):
    for path in tqdm(splits[key]):
        src = os.path.join('food-101', 'images', path + '.jpg')
        c, n = path.split('/')
        dst = os.path.join('food-101', key, c, n + '.jpg')
        copy2(src, dst)

copy('train')
copy('valid')
