from datasets import CSdatasets, CTXdatasets, ADEdatasets, COCOdatasets
from mypath import Path


def build_dataset(dataset, split='val', data_dir='val', crop_size=(512,512), scale=False, mirror=False, brightness=False, ignore_label=255, balance=0, longsize=-1, shortsize=-1, data_para={}):
    root, list_path = Path.data_dir(dataset,data_dir)
    return eval(dataset+'datasets').DataSet(root, list_path, split=split, crop_size=crop_size, scale=scale, mirror=mirror, brightness=brightness, ignore_label=ignore_label, balance=balance, longsize=longsize, shortsize=shortsize, **data_para)
