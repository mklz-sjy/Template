#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from infers.simple_mnist_infer import SimpleMnistInfer
from utils.config_utils import get_test_args, get_config_from_json
from data_utils.simple_mnist_dl import get_test_data
from torch.utils.data import DataLoader
from models.simple_mnist_model import BuildModel
import torch


def test_main():
    print('[1/5] 解析配置...')
    parser = None
    config = None
    try:
        args, parser = get_test_args()
        config, _ = get_config_from_json(args.cfg)
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] 参考: python main_train.py -c configs/simple_mnist_config.json')
        exit(0)
    print('[1/5] 解析配置--Done')

    print('[2/5] 加载数据...')
    test_dataset = get_test_data()
    test_loader = DataLoader(test_dataset, config.batch_size, shuffle=False, num_workers=config.workers)
    print('[2/5] 加载数据--Done')

    print('[3/5] 加载网络...')
    model = BuildModel(config)
    model.to(config.device)
    model.load_state_dict(torch.load(args.weights))
    print('[3/5] 加载网络--Done')

    print('[4/5] 预测数据...')
    infer = SimpleMnistInfer(
        test_loader,
        model=model,
        config=config)
    infer.eval()
    print('[4/5] 预测数据--Done')

    print('[5/5] 预测完成--Done')

if __name__ == '__main__':
    test_main()
