# DL Project Template

[深度学习Pytorch工程模板]

分块封装，简化加载数据、构建网络、训练模型和预测样本的流程

## 使用方式

下载工程

```text
git clone https://github.com/mklz-sjy/Template.git
```

创建和激活虚拟环境

```text
conda create -n [env-name] python=3.8
conda activate [env-name]
```

安装Python依赖库

```text
pip install -r requirements.txt
```

开发流程

1. 定义自己的数据加载类，继承DatasetBase；
2. 定义自己的网络结构类，继承ModelBase；
3. 定义自己的模型训练类，继承TrainerBase；
4. 定义自己的样本预测类，继承InferBase；
4. 定义自己的配置文件，写入实验的相关参数；

执行训练模型和预测样本操作。

## 示例工程

识别[MNIST](http://yann.lecun.com/exdb/mnist/)库中手写数字，工程``simple_mnist``

训练：

```text
python main_train.py -c configs\simple_mnist_config.json
```

预测：

```text
python main_test.py -c configs\simple_mnist_test_config.json -w output\simple_mnist\checkpoints\09_000000.pth
```

## 工程架构

### 框架图

![架构](doc/arc.png)

### 文件夹结构

```text
├── bases
│   ├── data_base.py             - 数据加载基类
│   ├── infer_base.py                   - 预测样本（推断）基类
│   ├── model_base.py                   - 网络结构（模型）基类
│   ├── trainer_base.py                 - 训练模型基类
├── configs                             - 配置文件夹
│   └── simple_mnist_config.json        -配置文件
├── data                                - 数据文件夹 
│   └──MNIST                            -MNIST数据集
├── data_utils                        - 数据加载文件夹
│   ├── __init__.py
│   ├── simple_mnist_dl.py
├── infers                              - 推断文件夹
│   ├── __init__.py
│   ├── simple_mnist_infer.py
├── models                              - 网络结构文件夹
│   ├── __init__.py
│   ├── simple_mnist_model.py
├── output                         - 实验数据文件夹
│   └── simple_mnist                    - 实验名称
│       ├── checkpoints                 - 存储的参数
│       │   └── 09_ACC_0.893400.pth
│       ├── images                      - 存储结果图
│       └── logs                        - 日志，如TensorBoard
│           └── events.out.tfevents.1524034653.wang
├── trainers                            - 训练模型文件夹
│   ├── __init__.py
│   ├── simple_mnist_trainer.py
└── utils                               - 工具文件夹
    ├── __init__.py
    ├── config_utils.py                 - 配置工具类
    ├── logger.py                       - 损失记录类
    ├── np_utils.py                     - NumPy工具类
    ├── utils.py                        - 其他工具类

├── main_test.py                        - 预测样本入口
├── main_train.py                       - 训练模型入口
├── requirements.txt                    - 依赖库
```

## 主要组件

### DataLoader

操作步骤：

1. 创建自己的加载数据类，继承DatasetBase基类；
2. 覆写``__init__``、``__getitem__``和``__len__``，返回训练和测试数据；

### Model

操作步骤：

1. 创建自己的网络结构类，继承ModelBase基类；
2. 覆写``__init__``和``forward``，创建网络结构；

### Trainer

操作步骤：

1. 创建自己的训练类，继承TrainerBase基类；
2. 参数：train_loader, val_loader, 网络结构model、配置config；
3. 覆写``val_per_epoch()``，``compute_metrics``和``compute_loss``函数；

### Infer

操作步骤：

1. 创建自己的预测类，继承InferBase基类；
2. 覆写``eval()``，完善验证功能；
3. 覆写``compute_metrics()``，提供指标计算功能；


### Config

定义在模型训练过程中所需的参数，JSON格式，支持：学习率、Epoch、Batch等参数。

### Main

训练：

1. 创建配置文件config；
2. 创建数据加载类dataloader；
3. 创建网络结构类model；
4. 创建训练类trainer，参数是训练和测试数据、模型；
5. 执行训练类trainer的train()；

预测：

1. 创建配置文件config；
2. 创建数据加载类dataloader；
3. 创建网络结构类model并加载权重参数；
4. 执行预测类infer的eval()；

## 感谢

参考[DL-Project-Template](https://github.com/SpikeKing/DL-Project-Template)





