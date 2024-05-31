import os
import os.path as osp
import json
import numpy as np
import torch

from .Base import BaseDataSet
from utils.pyt_utils import load_dict


class DataSet(BaseDataSet):
    def __init__(self, root, list_path, max_iters=None, split='train', crop_size=(321, 321), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True,brightness=True, ignore_label=255, balance=0, longsize=-1, shortsize=-1, **kwargs):
        super(DataSet, self).__init__(split=split,crop_size=crop_size, mean=mean, std=std, scale=scale, mirror=mirror, brightness=brightness, ignore_label=ignore_label, balance=balance, **kwargs)
        self.num_classes = 150
        self.long_size = longsize
        self.short_size = shortsize 
        self.root = root
        self.list_path = list_path
        self.class_weights = None
        self.cmap_labels = np.array([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
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
                           [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                           [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                           [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                           [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                           [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                           [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                           [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                           [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                           [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                           [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                           [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                           [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                           [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                           [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                           [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                           [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                           [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                           [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                           [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                           [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                           [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                           [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                           [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                           [102, 255, 0], [92, 0, 255]])
        
        self.img_ids = [json.loads(x.rstrip()) for x in open(list_path, 'r')]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            image_path, label_path = item['fpath_img'], item['fpath_segm']
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        if self.resample:
            self.class_files = load_dict(osp.join(os.path.dirname(self.list_path), 'label_index_ADE.pkl'))
        print('{} images are loaded!'.format(len(self.img_ids)))
    
    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            label_copy[label_copy == self.ignore_label] = -1
            label_copy = label_copy + 1
        else:
            label_copy = label_copy - 1
            label_copy[label_copy < 0] = self.ignore_label
        return label_copy
