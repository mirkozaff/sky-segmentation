# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from PIL import Image
from glob import glob
import numpy as np
import scipy.io as sio
from utils import palette
import torch
import os
import cv2
from tqdm import tqdm

class CocoStuff10k(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.COCO_palette
        super(CocoStuff10k, self).__init__(**kwargs)

    def _set_files(self):
        file_list = sorted(glob(os.path.join(self.root, 'images', '*.jpg')))
        self.files = [os.path.basename(f).split('.')[0] for f in file_list]
        temp = []
        n_images = 0
        for image_id in tqdm(self.files):
            label_path = os.path.join(self.root, 'annotations', image_id + '.png')
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # If image contains sky or clouds pixels use it
            if (np.sum(label) > 1):
                n_images += 1                    
                temp.append(image_id)
        print(f"#image: {n_images}")
        # Update file list
        self.files = temp[:]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, 'images', image_id + '.jpg')
        label_path = os.path.join(self.root, 'annotations', image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label[label == 255] = 1
        
        return image, label, image_id

def get_parent_class(value, dictionary):
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res

class COCO_agg(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False, inference = False):

        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]
        
        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'inference': inference
        }

        self.dataset = CocoStuff10k(**kwargs)
        print(self.dataset)

        super(COCO_agg, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

