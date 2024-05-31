import os
import os.path as osp
import numpy as np
import torch

from .Base import BaseDataSet
from utils.pyt_utils import load_dict


class DataSet(BaseDataSet):
    def __init__(self, root, list_path, max_iters=None, split='train', crop_size=(321, 321), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True,brightness=True, ignore_label=255, balance=0, longsize=-1, shortsize=-1, **kwargs):
        super(DataSet, self).__init__(split=split,crop_size=crop_size, mean=mean, std=std, scale=scale, mirror=mirror, brightness=brightness, ignore_label=ignore_label, balance=balance, **kwargs)
        self.num_classes = 19
        self.long_size = longsize
        self.short_size = shortsize
        self.root = root
        self.list_path = list_path
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()
        self.label_total_sum = np.array([2.03641652e+09, 3.36090793e+08, 1.26063612e+09, 3.61994980e+07,
                               4.84541660e+07, 6.77895060e+07, 1.14770880e+07, 3.04481930e+07,
                               8.79783988e+08, 6.39495360e+07, 2.21979646e+08, 6.73264240e+07,
                               7.46316200e+06, 3.86328286e+08, 1.47723280e+07, 1.29902900e+07,
                               1.28639550e+07, 5.44915200e+06, 2.28612330e+07])
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        self.cmap_labels = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                            [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                            [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

        if split == 'test':
            self.img_ids = [i_id.strip().split()[0] for i_id in open(list_path)]
            self.files = []
            for image_path in self.img_ids:
                name = osp.splitext(osp.basename(image_path))[0]
                img_file = osp.join(self.root, image_path)
                self.files.append({
                    "img": img_file,
                    "name": name
                })
        else:
            self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
            if not max_iters == None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.files = []
            for item in self.img_ids:
                image_path, label_path = item
                name = osp.splitext(osp.basename(label_path))[0]
                img_file = osp.join(self.root, image_path)
                label_file = osp.join(self.root, label_path)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
            if self.resample:
                if len(self.img_ids) == 2975:
                    self.class_files = load_dict(osp.join(os.path.dirname(self.list_path),'label_index_CS.pkl'))
                elif len(self.img_ids) == 3475:
                    self.class_files = load_dict(osp.join(os.path.dirname(self.list_path), 'label_index_CStest.pkl'))
        print('{} images are loaded!'.format(len(self.img_ids)))
        
    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy
    
