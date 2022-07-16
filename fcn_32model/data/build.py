# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import imp
from torch.utils import data

from .datasets.voc import VocSegDataset
from .transforms import build_transforms


def build_dataset(cfg, transforms, is_train=True):
    datasets = VocSegDataset(cfg, is_train, transforms)
    return datasets


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)

    datasets = build_dataset(cfg, transforms, is_train)

    #print(datasets.data_list, datasets.label_list)
    # from PIL import Image
    # img = Image.open(datasets.data_list[0])
    # import numpy as np
    # img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    # print(img.shape)

    # import cv2
    # img = cv2.imread(datasets.data_list[0])
    # cv2.imshow('title', img)
    # cv2.waitKey(0)


    # _, (img,_) = next(enumerate(datasets))
    # #print(img.numpy().transpose(1,2,0))
    # import cv2
    # cv2.imshow('title', img.numpy().transpose(1,2,0)[:,:,list([2,0,1])])
    # cv2.waitKey(0)
    
    # while 1:
    #     pass

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=1, shuffle=shuffle, num_workers=0, pin_memory=False
    )

    # _, (img,_) = next(enumerate(data_loader))
    # print(img.numpy().shape)
    # img = img[0]
    # import cv2
    # cv2.imshow('title', img.numpy().transpose(1,2,0)[:,:,list([2,0,1])])
    # cv2.waitKey(0)
    
    # while 1:
    #     pass

    return data_loader
