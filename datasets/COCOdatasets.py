import os
import os.path as osp
import numpy as np
import torch

from .Base import BaseDataSet
from utils.pyt_utils import load_dict


class DataSet(BaseDataSet):
    def __init__(self, root, list_path, max_iters=None, split='train', crop_size=(321, 321), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True,brightness=True, ignore_label=255, balance=0, longsize=-1, shortsize=-1, **kwargs):
        super(DataSet, self).__init__(split=split,crop_size=crop_size, mean=mean, std=std, scale=scale, mirror=mirror, brightness=brightness, ignore_label=ignore_label, balance=balance, **kwargs)
        self.num_classes = 171
        self.long_size = longsize
        self.short_size = shortsize 
        self.root = root
        self.list_path = list_path
        self.class_weights = None
        self.cmap_labels = np.array([[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
               [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
               [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
               [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
               [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
               [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
               [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
               [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
               [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
               [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
               [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
               [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
               [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
               [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
               [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
               [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
               [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
               [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
               [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
               [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
               [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
               [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
               [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
               [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
               [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
               [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
               [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
               [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
               [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
               [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
               [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
               [64, 192, 96], [64, 160, 64], [64, 64, 0]])
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s.jpg" % name)
            label_file = osp.join(self.root, "annotations/%s_labelTrainIds.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        if self.resample:
            self.class_files = load_dict(osp.join(os.path.dirname(self.list_path), 'label_index_COCO.pkl'))
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
    