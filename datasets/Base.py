import numpy as np
import random
import torch
from torch.nn import functional as F
import cv2
from torch.utils import data
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate

from utils.edge_utils import onehot_to_multiclass_edges, mask_to_onehot


class BaseDataSet(data.Dataset):
    def __init__(self, split='train', crop_size=(321, 321), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True,brightness=True, ignore_label=255, balance=0, **kwargs):
        self.split = split
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.std = std
        self.scale = scale
        self.is_mirror = mirror
        self.brightness = brightness
        self.ignore_label = ignore_label
        self.balance = balance
        self.resample = kwargs.get('resample',False)
        if self.balance==2:
            self.beta = kwargs.get('beta',0.9999)
        
    def __len__(self):
        if self.resample:
            return int(self.class_files['label_f'].max()*self.num_classes)
        else:
            return len(self.files)
        
    def pre_processing(self, epoch, max_epoch):
        if self.resample:
            self.gen_index(epoch)

    def gen_index(self, seed=0):
#         np.random.seed(seed)
        length = np.int(self.class_files['label_f'].max())
        self.file_index = []
        self.class_index = []
        for i in np.arange(self.num_classes):
            len_i = len(self.class_files[str(i)])
            ind = np.arange(len_i).tolist()*(length//len_i)
            last = length % len_i
#             ind = ind+np.random.choice(np.arange(len_i),replace=False,size=last).tolist()
            ind = ind+random.sample(np.arange(len_i).tolist(),last)
            self.file_index = self.file_index + ind
            self.class_index = self.class_index + (np.ones(len(ind))*i).astype(int).tolist()
        if (dist.is_available() and dist.is_initialized()):
            self.file_index = torch.from_numpy(np.asarray(self.file_index)).cuda()
            self.class_index = torch.from_numpy(np.asarray(self.class_index)).cuda()
            dist.broadcast(self.file_index, 0)
            dist.broadcast(self.class_index, 0)
            self.file_index = self.file_index.cpu().numpy().tolist()
            self.class_index = self.class_index.cpu().numpy().tolist()
        print('created balance dataset')
        
    def get_datafile(self, index):
        img_meta = {"idx":index}
        if self.resample:
            class_ = self.class_index[index]
            index = self.class_files[str(class_)][self.file_index[index]]['idx']
            datafile = self.files[index]
            img_meta["index"] = index
            img_meta["class"] = class_
        else:
            datafile = self.files[index]
        img_meta["name"]=datafile["name"]
        return datafile, img_meta
        
    def get_label(self, label, img_meta):
        if self.balance>0:
            labels = {'ori': label.copy()}
            label_balance = label.copy()
            label_balance[label == self.ignore_label] = self.num_classes
            class_num = np.bincount(label_balance.reshape(-1), minlength=self.num_classes + 1)[:-1]
            if self.balance == 1:
                weight_class = 1 / (class_num+1)
            elif self.balance == 2:
                weight_class = (1 + 1e-8 - self.beta ** class_num[img_meta["class"]]) / (1 + 1e-8 - self.beta ** class_num)
            weight_class = np.clip(weight_class, 0.0, 1.0)
            weight_class = np.append(weight_class, 0)
            weight = weight_class[label_balance]
            labels['weight'] = weight
        else:
            labels = label.copy()
        return labels
    
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def generate_scale_label(self, image, label, img_meta):
        f_scale = 0.5 + random.randint(0, 15) / 10.0
        if self.long_size > 0:
            long_size = np.int(self.long_size * f_scale+0.5)
            h, w = image.shape[:2]
            f_scale = long_size*1.0/max(h,w)
        elif self.short_size > 0:
            short_size = np.int(self.short_size * f_scale+0.5)
            h, w = image.shape[:2]
            f_scale = short_size*1.0/min(h,w)
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, img_meta
    
    def random_brightness(self, img):
        if random.random() < 0.5:
            return img
        self.shift_value = 10
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def random_contrast(self, img):
        if random.random() < 0.5:
            return img
        self.contrast_lower = 0.75
        self.contrast_upper = 1.25
        img = img.astype(np.float32)
        alpha = random.uniform(self.contrast_lower, self.contrast_upper)
        img = img * alpha
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def random_saturation(self, img):
        if random.random() < 0.5:
            return img
        self.saturation_lower = 0.75
        self.saturation_upper = 1.25
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        alpha = random.uniform(self.saturation_lower, self.saturation_upper)
        hsv[:, :, 1] = hsv[:, :, 1] * alpha
        hsv[:, :, 1] = np.around(hsv[:, :, 1])
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img
    
    def random_contrast(self, img):
        if random.random() < 0.5:
            return img
        self.contrast_lower = 0.75
        self.contrast_upper = 1.25
        img = img.astype(np.float32)
        alpha = random.uniform(self.contrast_lower, self.contrast_upper)
        img = img * alpha
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def random_saturation(self, img):
        if random.random() < 0.5:
            return img
        self.saturation_lower = 0.75
        self.saturation_upper = 1.25
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        alpha = random.uniform(self.saturation_lower, self.saturation_upper)
        hsv[:, :, 1] = hsv[:, :, 1] * alpha
        hsv[:, :, 1] = np.around(hsv[:, :, 1])
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img

    def random_hue(self, img):
        if random.random() < 0.5:
            return img
        self.hue_delta = 18
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-self.hue_delta, self.hue_delta)) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def crop_img(self, img, label, img_meta):
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = img, label

        h_off,w_off = self.get_crop_location(label_pad, img_meta)
        img = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.int)
        return img, label, img_meta
    
    def get_crop_location(self,label, img_meta):
        img_h, img_w = label.shape
        if self.resample:
            label_i = (label==img_meta["class"]).astype(np.uint8)
            nums_i, label_i = cv2.connectedComponents(label_i, connectivity=8)
            if nums_i>=2:
                n = random.randint(1,nums_i-1)
                h, w = np.where(label_i==n)
                n = random.randint(0,len(h)-1)
                h_off = h[n]-self.crop_h//2-random.randint(-self.crop_h//4, self.crop_h//4)
                w_off = w[n]-self.crop_w//2-random.randint(-self.crop_w//4, self.crop_w//4)
            else:
                h_off = random.randint(0, img_h - self.crop_h)
                w_off = random.randint(0, img_w - self.crop_w)
        else:
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
        h_off = np.clip(h_off, 0, img_h - self.crop_h)
        w_off = np.clip(w_off, 0, img_w - self.crop_w)
        return h_off,w_off
    
    def __getitem__(self, index):
        datafile, img_meta = self.get_datafile(index)
        image = cv2.imread(datafile["img"], cv2.IMREAD_COLOR)
        img_meta["size"] = np.array(image.shape)
        if self.split=='test':
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return {"img":image.copy(), "img_meta":img_meta}
        else:
            label = cv2.imread(datafile["label"], cv2.IMREAD_GRAYSCALE)
            label = self.id2trainId(label)
            if self.split=='val':
                image = self.input_transform(image)
                image = image.transpose((2, 0, 1))
                return {"img":image.copy(), "label":label.copy(), "img_meta":img_meta}
            else:
                if self.scale:
                    image, label, img_meta = self.generate_scale_label(image, label, img_meta)
                if self.brightness:
                    image = self.random_brightness(image)
                    mode = random.randint(0, 1)
                    if mode == 1:
                        image = self.random_contrast(image)
                    image = self.random_saturation(image)
                    image = self.random_hue(image)
                    if mode == 0:
                        image = self.random_contrast(image)
                image = self.input_transform(image)
                image, label, img_meta = self.crop_img(image, label, img_meta)
                #image = image[:, :, ::-1]  # change to BGR
                image = image.transpose((2, 0, 1))
                if self.is_mirror:
                    flip = random.randint(0, 1) * 2 - 1
                    image = image[:, :, ::flip]
                    label = label[:, ::flip]
                label = self.get_label(label, img_meta)

                return {"img":image.copy(), "label":label, "img_meta":img_meta}
            
            
def base_convert(batch):
    img_meta = []
    for data in batch:
        img_meta.append(data.pop("img_meta"))
    out_data = default_collate(batch)
    out_data["img_meta"] = img_meta
    return out_data
        