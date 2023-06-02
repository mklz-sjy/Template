# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from torch.utils.data import Dataset

class DatasetBase(Dataset):
    """
        数据加载的基类
    """
    def __init__(self, config):
        super(DatasetBase, self).__init__()
        self.config = config
        # 根据具体的业务逻辑读取全部数据路径作为加载数据的索引

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

