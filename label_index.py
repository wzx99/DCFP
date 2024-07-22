import sys
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from six.moves import cPickle as pickle

from datasets import build_dataset

# STEP = 100000


def get_parser():
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--dataset", type=str, default='CS',
                        help="choose dataset.")
    parser.add_argument("--save-dir", type=str)
    return parser


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def label_index():
    parser = get_parser()
    args = parser.parse_args()

    dataset = build_dataset(args.dataset, split='train', data_dir='train')

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(dataset.__len__()), file=sys.stdout,
                bar_format=bar_format)
    # iii = 0
    index_ = {}
    for i in np.arange(dataset.num_classes):
        index_[str(i)] = []
    for idx in pbar:
        datafiles = dataset.files[idx]
        tmp_dict = {'idx':idx, 'name':datafiles['name']}

        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = dataset.id2trainId(label)
        label[label==dataset.ignore_label] = dataset.num_classes

        count = np.bincount(label.reshape(-1), minlength=dataset.num_classes + 1)[:-1]
        label_index = np.where(count>0)[0]
        for i in label_index:
            index_[str(i)].append(tmp_dict)

        print_str = ' Iter{}/{}'.format(idx + 1, dataset.__len__())
        pbar.set_description(print_str, refresh=False)

        # iii = iii + 1
        # if iii >= STEP:
        #     break

    index_['label_f'] = np.zeros(dataset.num_classes)
    for i in np.arange(dataset.num_classes):
        index_['label_f'][i] = len(index_[str(i)])
    save_path = os.path.join(args.save_dir, 'label_index_'+args.dataset+'.pkl')
    save_dict(index_, save_path)


if __name__ == '__main__':
    label_index()
