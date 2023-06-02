# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import argparse
import json

import os
from bunch import Bunch

from utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    """
    with open(json_file, 'r') as config_file:
        content = config_file.read()
        config_dict = json.loads(content)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict


def process_config(json_file):
    """
    解析Json文件
    :param json_file: 配置文件
    :return: 配置类
    """
    config, _ = get_config_from_json(json_file)
    config.tb_dir = os.path.join(config.model_dir, config.exp_name, "logs/")  # 日志
    config.cp_dir = os.path.join(config.model_dir, config.exp_name, "checkpoints/")  # 模型
    config.img_dir = os.path.join(config.model_dir, config.exp_name, "images/")  # 网络
    mkdir_if_not_exist(os.path.join(config.model_dir, config.exp_name))
    mkdir_if_not_exist(config.tb_dir)
    mkdir_if_not_exist(config.cp_dir)
    mkdir_if_not_exist(config.img_dir)
    return config


def get_train_args():
    """
    添加训练参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c','--cfg',
        type=str,
        default='',
        required=True,
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser


def get_test_args():
    """
    添加测试路径
    :return: 参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c','--cfg',
        type=str,
        default='',
        required=True,
        help='add a configuration file')
    parser.add_argument(
        '-w','--weights',
        type=str,
        default='',
        required=True,
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser
