import os
import os.path as osp
import numpy as np
import torch

from .Base import BaseDataSet
from utils.pyt_utils import load_dict


class DataSet(BaseDataSet):
    def __init__(self, root, list_path, max_iters=None, split='train', crop_size=(321, 321), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True,brightness=True, ignore_label=255, balance=0, longsize=-1, shortsize=-1, **kwargs):
        super(DataSet, self).__init__(split=split,crop_size=crop_size, mean=mean, std=std, scale=scale, mirror=mirror, brightness=brightness, ignore_label=ignore_label, balance=balance, **kwargs)
        self.num_classes = 59
        self.long_size = longsize
        self.short_size = shortsize 
        self.root = root
        self.list_path = list_path
        self.class_weights = None
        self.cmap_labels = np.array([[180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s.jpg" % name)
            label_file = osp.join(self.root, "labels/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        if self.resample:
            self.class_files = load_dict(osp.join(os.path.dirname(self.list_path), 'label_index_CTX.pkl'))
        print('{} images are loaded!'.format(len(self.files)))
    
    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            label_copy[label_copy == self.ignore_label] = -1
            label_copy = label_copy + 1
        else:
            label_copy = label_copy - 1
            label_copy[label_copy < 0] = self.ignore_label
        return label_copy
    