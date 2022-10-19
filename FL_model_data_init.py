#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Fl
# @Time      :2022/5/1 16:11
# @Author    :Ouyang Bin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader


def model_init(data_name):
    if data_name == 'adult':
        model = Net_adult()
    return model


# 108->50->10->2
class Net_adult(nn.Module):
    def __init__(self):
        super(Net_adult, self).__init__()
        self.fc1 = nn.Linear(108, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.relu(x)
        return x


# 卷积网络
class All_CNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(All_CNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            # 防止过拟合
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            # 平均池化，默认向下取整
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output


# 自定义卷积类
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:  # 卷积
            # model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            # padding=padding)]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:  # 转置卷积
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            # 归一化
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        # 激活函数 ReLu
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


# 不区分参数的占位符标识运算符 在增减网络过程中，可以使得整个网络层数据不变，便于迁移权重数据
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# 保存batch维数，其他维压平
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


"""Function: load data"""


def data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)

    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    # 构建测试数据加载器   **kwargs允许你将不定长度的键值对, 作为参数传递给一个函数  drop_last=True则：若最后一个epoch的数据不完整则删除
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, drop_last=True, **kwargs)

    # 将数据按照训练的trainset，均匀的分配成N-client份，所有分割得到dataset都保存在一个list中
    split_index = [int(trainset.__len__() / FL_params.N_client)] * (
            FL_params.N_client - 1)  # split_index=[6512, 6512, 6512, 6512]
    split_index.append(
        int(trainset.__len__() - int(trainset.__len__() / FL_params.N_client) * (FL_params.N_client - 1)))
    # split_index = [6512, 6512, 6512, 6512, 6513]
    client_dataset = torch.utils.data.random_split(trainset, split_index)  # 随机分成6512，…… ，6513五个数据集

    # 将全局模型复制N-client次，然后构建每一个client模型的优化器，参数记录
    client_loaders = []
    for ii in range(FL_params.N_client):
        client_loaders.append(
            DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=False, drop_last=True, **kwargs))
        '''
        By now，我们已经将client用户的本地数据区分完成，存放在client_loaders中。每一个都对应的是某一个用户的私有数据
        '''

    return client_loaders, test_loader


# 返回不同的训练集与测试集
def data_set(data_name):
    # 数据集不是mnist,purchase,adult,cifar10,报错
    if data_name not in ['mnist', 'purchase', 'adult', 'cifar10']:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')

    if data_name == 'adult':
        # load data
        file_path = "./data/adult/"
        data1 = pd.read_csv(file_path + 'adult.data', header=None)  # 读取adult.data里的数据 （32561， 15）
        data2 = pd.read_csv(file_path + 'adult.test', header=None)  # （16281， 15）
        data2 = data2.replace(' <=50K.', ' <=50K')  # 将数据集中的<=50k.替换为<=50k (数据集中有'<=50k.'?)
        data2 = data2.replace(' >50K.', ' >50K')
        train_num = data1.shape[0]
        # data.shape=(48842, 15) 48842条数据，15列
        data = pd.concat([data1, data2])  # 将data1与data2上下拼接（默认axis=0） join参数控制是否为公共部分（’inner‘）

        # data transform: str->int
        data = np.array(data, dtype=str)
        # 第15列为y 标签
        labels = data[:, 14]
        le = LabelEncoder()

        # 这两步使用.fit_transform()一步到位
        le.fit(labels)  # 编码为0和1（只有两种情况）
        labels = le.transform(labels)  # label已经编码成功

        # data.shape=(48842, 14)
        data = data[:, :-1]  # 除去y后的数据集

        # 把类别进行数字化--LabelEncoder
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]  # 工作类别；受教育程度；婚姻状况；职业；社会角色；种族；性别；国籍
        # categorical_names = {}

        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])  # 将选中的列（特征）进行编码 这些列为字符串类型
            # categorical_names[feature] = le.classes_

        data = data.astype(float)  # 转换为float类型

        n_features = data.shape[1]  # 特征数量
        numerical_features = list(
            set(range(n_features)).difference(set(categorical_features)))  # 全集与catagorical_features的差集
        # numerical_features = [0, 2, 4, 10, 11, 12]
        # 数据归一化(0-1)
        # 格式依旧是48842行、14列
        for feature in numerical_features:
            scaler = MinMaxScaler()  # X-X.min(axis=0) / X.max(axis=0)-X.min(axis=0) * (max-min) - min
            sacled_data = scaler.fit_transform(data[:, feature].reshape(-1, 1))  # 进行归一化
            data[:, feature] = sacled_data.reshape(-1)  # 没必要进行

        # 把数据进行01编码--OneHotLabel，只作用于[1, 3, 5, 6, 7, 8, 9, 13]列，其他列不变
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features), ],
            remainder='passthrough')  # 名称，对象，列   remainder='passthrough'代表只想转换分类数据并不改变数字列
        #  sparse=False代表不为稀疏格式，无需.toarray(),若为True则要
        oh_data = oh_encoder.fit_transform(data)  # 独热编码
        # 生成数据格式：48842*108，其中[0, 2, 4, 10, 11, 12]在108列的最后6列，其余列进行了0-1编码

        xx = oh_data
        yy = labels

        yy = np.array(yy)  # 可删去

        xx = torch.Tensor(xx).type(torch.FloatTensor)
        yy = torch.Tensor(yy).type(torch.LongTensor)
        xx_train = xx[0:data1.shape[0], :]
        xx_test = xx[data1.shape[0]:, :]
        yy_train = yy[0:data1.shape[0]]
        yy_test = yy[data1.shape[0]:]

        # trainset = Array2Dataset(xx_train, yy_train)
        # testset = Array2Dataset(xx_test, yy_test)
        trainset = TensorDataset(xx_train, yy_train)  # 将xx_train与yy_train左右拼接，trainset = xx-train, yy_train
        testset = TensorDataset(xx_test, yy_test)

    return trainset, testset
