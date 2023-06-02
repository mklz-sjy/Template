# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from torchvision.datasets import MNIST
import numpy as np
from bases.data_base import DatasetBase
from torchvision.transforms import Compose,ToTensor,Normalize

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def get_train_data():
    train_dataset = MNIST('./data/', train=True, download=True,
                               transform=Compose([
                                   ToTensor(),
                                   Normalize((0.1307,), (0.3081,))
                               ]))
    return train_dataset

def get_test_data():
    test_dataset = MNIST('./data/', train=False, download=True,
                               transform=Compose([
                                   ToTensor(),
                                   Normalize((0.1307,), (0.3081,))
                               ]))
    return test_dataset