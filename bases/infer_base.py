#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from utils.logger import Logger
import os
from torch.autograd import Variable

class InferBase(object):
    """
    推断基类
    """
    def __init__(self, test_loader, model, config):
        self.config = config  # 配置
        self.model = model
        self.test_loader = test_loader
        self.logger = Logger(config)

        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

    def eval(self):
        self.model.eval()
        for i, data in enumerate(self.test_loader):
            img, pred, label = self.step(data)

            print('Test: {}/{} '.format(i, len(self.test_loader)))

    def step(self, data):
        img, label = data
        # warp input
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(img)
        return img, label, pred

    def compute_metrics(self, pred, gt):
        # you can call functions in metrics.py
        raise NotImplementedError
