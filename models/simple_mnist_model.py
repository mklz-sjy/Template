# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import torch.nn as nn
import torch.nn.functional as F


from bases.model_base import ModelBase

class SimpleMnistModel(ModelBase):
    """
    SimpleMnist模型
    """

    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.softmax(x)

# import timm
# model = timm.create_model(model_name=args.net, pretrained=True, num_classes=3)
def BuildModel(config):
    if config.exp_name == 'simple_mnist':
        model = SimpleMnistModel(config)
    elif config.exp_name == 'resnet18':
        import torchvision
        model = torchvision.models.resnet18(num_classes=config.num_classes)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model