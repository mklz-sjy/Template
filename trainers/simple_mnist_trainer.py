# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from bases.trainer_base import TrainerBase
from sklearn.metrics import accuracy_score
import torch

class SimpleMnistTrainer(TrainerBase):
    """
    训练器基类
    """
    def __init__(self, train_loader, val_loader, model, config):
        super(SimpleMnistTrainer, self).__init__(train_loader, val_loader, model, config)

    def val_per_epoch(self, epoch):
        self.model.eval()
        # 不同任务此处评价方式需要调整
        preds = []
        labels = []
        for i, data in enumerate(self.val_loader):
            img, pred, label = self.step(data)
            preds.extend(torch.argmax(pred, axis=1).cpu().numpy())
            labels.extend(label.cpu().numpy())

            loss = self.compute_loss(pred, label)
            self.logger.record_scalar('val_loss', loss)

            # monitor training progress
            print('Val: Epoch_{}/{} Iteration_{}/{} Loss {}'.format(epoch, self.config.epochs, i, len(self.val_loader),
                                                                    loss))

        metrics = self.compute_metrics(preds, labels)
        self.logger.writer.add_scalar('ACC', metrics, epoch)
        print('Val: Epoch_{}/{} ACC {}'.format(epoch, self.config.epochs, metrics))
        return {'ACC':metrics}

    def compute_metrics(self, preds, gts):
        return accuracy_score(gts, preds)


    def compute_loss(self, pred, gt):
        loss = torch.nn.functional.cross_entropy(pred,gt)
        return loss