# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""

import torch.nn as nn

class ModelBase(nn.Module):
    """
    模型基类
    """

    def __init__(self, config):
        super(ModelBase,self).__init__()
        self.config = config  # 配置
        self.model = None  # 模型

    def forward(self,x):
        """
        构建模型
        """
        raise NotImplementedError