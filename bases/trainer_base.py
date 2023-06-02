# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from utils.logger import Logger
import os
import torch
import numpy as np
import random
from torch.autograd import Variable

class TrainerBase(object):
    """
    训练器基类
    """
    def __init__(self, train_loader, val_loader, model, config):
        self.model = model  # 模型
        self.config = config  # 配置
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = Logger(config)#记录

        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
        # fix the seed for reproducibility
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)  # 固定随机数

        params = [p for p in self.model.parameters() if p.requires_grad]
        if config.optim == "Adam":
            self.optimizer = torch.optim.Adam(params, config.lr,
                                          betas=(config.momentum, config.beta),
                                          weight_decay=config.weight_decay)
        elif config.optim == "SGD":
            self.optimizer = torch.optim.SGD(params, config.lr,
                                              momentum=config.momentum,
                                              weight_decay=config.weight_decay)
        else:
            raise NotImplemented


    def train(self):
        """
        训练逻辑
        """
        result_txt_path = self.config.model_dir + "/" + self.config.exp_name + "/" + 'result.txt'
        fd = open(result_txt_path, 'w')
        for epoch in range(self.config.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            metrics = self.val_per_epoch(epoch)
            self.logger.save_curves(epoch, fd)

            metrics_text = ''
            for key in metrics.keys():
                metrics_text += '{}_{:06f}'.format(key, metrics[key])
            fd.write("{}\n".format(metrics_text))

            if epoch == self.config.epochs-1:
                self.logger.save_check_point(self.model, epoch, metrics)
        fd.close()

    def train_per_epoch(self,epoch):
        self.model.train()

        for i, data in enumerate(self.train_loader):
            img, pred, label = self.step(data)

            # compute loss
            loss = self.compute_loss(pred, label)

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logger record
            self.logger.record_scalar('train_loss', loss)

            # monitor training progress
            print('Train: Epoch_{}/{} Iteration_{}/{} Loss {}'.format(epoch, self.config.epochs, i, len(self.train_loader), loss))

    def val_per_epoch(self,epoch):
        self.model.eval()
        #不同任务此处评价方式需要调整
        preds = []
        labels = []
        for i, data in enumerate(self.val_loader):
            img, pred, label = self.step(data)

            loss = self.compute_loss(pred, label)
            self.logger.record_scalar('val_loss', loss)
            # monitor training progress
            print('Val: Epoch_{}/{} Iteration_{}/{} Loss {}'.format(epoch, self.config.epochs, i, len(self.val_loader), loss))

        metrics_key = 'ACC'
        metrics = self.compute_metrics(preds,labels)
        return {metrics_key:metrics}

    def step(self, data):
        img, label = data
        # warp input
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(img)
        return img, pred, label

    def compute_metrics(self, pred, gt):
        raise NotImplemented

    def compute_loss(self, pred, gt):
        raise NotImplemented



