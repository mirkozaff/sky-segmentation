from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class TemplateDataset(BaseDataSet):
    """
    Load Dataset
    """
    def __init__(self, **kwargs):
        # Number of classes, if unlabelled is important then count it else set it as ignore_index
        self.num_classes = 5
        self.palette = palette.template_palette
        super(TemplateDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["training", "validation"]:
            self.image_dir = os.path.join(self.root, 'images', self.split)
            # load labels as grayscale images => (intensity=class_number, shape=(H,W,1))
            self.label_dir = os.path.join(self.root, 'annotations', self.split)
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
        else: raise ValueError(f"Invalid split name {self.split}")
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32) # set ignore_index to -1 or 255
        return image, label, image_id

class TemplateDataloader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        # to improve results calculate your own mean and std for your dataset
        self.MEAN = [0.5, 0.5, 0.5]
        self.STD = [0.25, 0.25, 0.25]

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
            'val': val
        }
                
        self.dataset = TemplateDataset(**kwargs)
        super(TemplateDataloader, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
