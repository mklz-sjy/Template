#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18

参考:
NumPy FutureWarning
https://stackoverflow.com/questions/48340392/futurewarning-conversion-of-the-second-argument-of-issubdtype-from-float-to
"""
from models.simple_mnist_model import BuildModel
from trainers.simple_mnist_trainer import SimpleMnistTrainer
from utils.config_utils import process_config, get_train_args
from data_utils.simple_mnist_dl import get_train_data, get_test_data
from torch.utils.data import DataLoader

def main_train():
    print('[1/5] 解析配置...')
    parser = None
    config = None
    try:
        args, parser = get_train_args()
        config = process_config(args.cfg)
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] 参考: python main_train.py -c configs/simple_mnist_config.json')
        exit(0)
    print('[1/5] 解析配置--Done')

    print('[2/5] 加载数据...')
    train_dataset = get_train_data()
    val_dataset = get_test_data()
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.workers)
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False, num_workers=config.workers)
    print('[2/5] 加载数据--Done')

    print('[3/5] 构造网络...')
    model = BuildModel(config)
    model.to(config.device)
    print('[3/5] 构造网络--Done')

    print('[4/5] 训练网络...')
    trainer = SimpleMnistTrainer(
        train_loader,
        val_loader,
        model=model,
        config=config)
    trainer.train()
    print('[4/5] 训练网络--Done')

    print('[5/5] 训练完成--Done')


if __name__ == '__main__':
    main_train()
