# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

from bases.infer_base import InferBase
from sklearn.metrics import accuracy_score
import torch


class SimpleMnistInfer(InferBase):
    def __init__(self, test_loader, model, config):
        super(SimpleMnistInfer, self).__init__(test_loader, model, config)

    def eval(self):
        self.model.eval()
        result_txt_path = self.config.model_dir + "/" + self.config.exp_name + "/" + 'test.txt'
        fd = open(result_txt_path, 'w')
        preds = []
        labels = []
        for i, data in enumerate(self.test_loader):
            img, label, pred = self.step(data)
            preds.extend(torch.argmax(pred, axis=1).cpu().numpy())
            labels.extend(label.cpu().numpy())

            print('Test: {}/{} '.format(i, len(self.test_loader)))
        metrics = self.compute_metrics(preds, labels)
        print('Test: ACC {}'.format(metrics))
        fd.write('Test: ACC {}'.format(metrics))
        fd.close()

    def compute_metrics(self, preds, gts):
        # you can call functions in metrics.py
        return accuracy_score(gts, preds)

