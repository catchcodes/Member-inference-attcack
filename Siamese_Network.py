#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Siamese_Network
# @Time      :2022/5/29 20:50
# @Author    :Ouyang Bin
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 定义孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),  # 四周镜像对称填充1行  e.g. 3x3 -> 5x5
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),  # inplace=True 覆盖原数据
            nn.BatchNorm2d(4),  # 归一化
            # nn.BatchNorm2d(4)中参数4为通道数 channels
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 10 * 21, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 2))

        self.dropout = nn.Dropout(p=0.5)

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# 定义损失函数 对比损失满足以下准则
# 近似样本之间的距离越小越好；不似样本之间的距离如果小于m，则通过互斥使其距离接近m。
# Loss = (1-Y)*1/2*l^2 + Y*1/2*max{0, m-l}^2
class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # 使用pairwise_distance计算欧式距离后，使用对比损失作为目标损失函数
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# 初始化孪生网络数据集  input_train,input_test为推断攻击距离数据
def Siamese_data_init(input_train, input_test):
    # 合并数据集
    input_X = list(torch.cat((input_train, input_test), 0))  # 80x11x21
    input_y = []
    # 0为相同类型，1为不同类型
    for _ in input_train:
        input_y.append(0)
    for _ in input_test:
        input_y.append(1)

    # 将数据集划分为训练集和测试集，返回X最外层为list，内层均为tensor；y为list类型   test数据集占60% 随机数种子为7
    X_train, X_test, y_train, y_test = train_test_split(input_X, input_y, test_size=0.6, shuffle=True, random_state=7)

    # 得到数据个数
    number_train = len(y_train)
    number_test = len(y_test)

    # 将训练集和测试集分为两组数据
    train_list = []
    i = 0
    for each_input in X_train:
        data1 = each_input[0].unsqueeze(0)  # 在第0维增加一维
        data2 = each_input[1].unsqueeze(0)
        for ii in range(len(each_input) - 2):
            data1 = torch.cat((data1, each_input[0].unsqueeze(0)), 0)
            data2 = torch.cat((data2, each_input[ii + 2].unsqueeze(0)), 0)

        train_list.append([data1, data2, y_train[i]])  # 两个输入
        i += 1

    test_list = []
    i = 0
    for each_input in X_test:
        data1 = each_input[0].unsqueeze(0)
        data2 = each_input[1].unsqueeze(0)
        for ii in range(len(each_input) - 2):
            data1 = torch.cat((data1, each_input[0].unsqueeze(0)), 0)
            data2 = torch.cat((data2, each_input[ii + 2].unsqueeze(0)), 0)

        test_list.append([data1, data2, y_test[i]])
        i += 1

    # 此时，data1、data2为tensor，label为int
    # 打印出来，便于数据处理
    # print("train_list:", train_list)
    # print("test_list:", test_list)
    DistanceTrain = []
    DistanceTest = []
    TrainSampleNumber = 0
    TestSampelNumber = 0
    for ReList in train_list:
        if ReList[2] == 0:
            DistanceTrain.append(ReList[0])
            TrainSampleNumber += 1
        if ReList[2] == 1:
            DistanceTest.append(ReList[0])
            TestSampelNumber += 1
    AddSampleNumber = min(TrainSampleNumber, TestSampelNumber)
    number_train += AddSampleNumber
    for i in range(AddSampleNumber):
        train_list.append([DistanceTrain[i], DistanceTest[i], 0])

    DistanceTrainInTest = []
    DistanceTestInTest = []
    TrainSampleNumberInTest = 0
    TestSampelNumberInTest = 0
    for ReList in test_list:
        if ReList[2] == 0:
            DistanceTrainInTest.append(ReList[0])
            TrainSampleNumberInTest += 1
        if ReList[2] == 1:
            DistanceTestInTest.append(ReList[0])
            TestSampelNumberInTest += 1
    AddSampleNumberInTest = min(TrainSampleNumberInTest, TestSampelNumberInTest)
    number_test += AddSampleNumberInTest
    for i in range(AddSampleNumberInTest):
        test_list.append([DistanceTrainInTest[i], DistanceTestInTest[i], 0])

    return train_list, test_list, number_train, number_test


# 训练孪生网络
def Siamese_function(input_train, input_test):
    # 孪生网络数据集初始化
    train_list, test_list, number_train, number_test = Siamese_data_init(input_train, input_test)

    # 初始化孪生网络、损失函数、优化器
    siamese_net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(siamese_net.parameters(), 0.001, betas=(0.9, 0.99))  # betas:用于计算梯度的平均和平方的系数(默认: (
    # 0.9, 0.999))

    print("siamese_net training start...")
    # 全局训练200轮
    similarity_train = []
    similarity_test = []

    siamese_net.train()

    for epoch in range(1, 201):
        running_loss = 0
        # distance0为距离列表第一行构成的矩阵、distance1为剩余行构成的矩阵
        # label为是否相似（即同为测试集/一个属于训练集一个属于测试集）
        for distance0, distance1, label in train_list:
            # print("distance0:", distance0)
            # print("distance1:", distance1)
            # print("label:", label)

            optimizer.zero_grad()

            # 将二维转化为四维，即为1*1*10*21
            # print(distance1.size())  # 10x21
            # print(distance0.size())  # 10x21
            distance0 = distance0.unsqueeze(0).unsqueeze(0)
            distance1 = distance1.unsqueeze(0).unsqueeze(0)

            output1, output2 = siamese_net(distance0, distance1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            running_loss += loss_contrastive.item()

        print("epoch: {}, loss: {}".format(epoch, running_loss))

        siamese_net.eval()
        # train集攻击模型测试
        if epoch % 1 == 0:
            score_label = []
            score_label_predict = []
            p_distance = []
            for distance0, distance1, label in train_list:
                distance0 = distance0.unsqueeze(0).unsqueeze(0)
                distance1 = distance1.unsqueeze(0).unsqueeze(0)

                output1, output2 = siamese_net(distance0, distance1)
                euclidean_distance = F.pairwise_distance(output1, output2)
                if euclidean_distance > 2:
                    euclidean_distance = 2
                p_distance.append(euclidean_distance)
                if euclidean_distance > 0.5:
                    # 认为不相似
                    label_predict = 1
                else:
                    label_predict = 0

                score_label.append(label)
                score_label_predict.append(label_predict)

            # print("p_distance:", p_distance)
            # print("label:", score_label)
            # print("label_predict:", score_label_predict)
            print("accuracy:", accuracy_score(score_label, score_label_predict))

            similarity_train.append(p_distance)

        # test集攻击模型测试
        if epoch % 1 == 0:
            score_labe = []
            score_labe_predict = []
            p_distanc = []
            for distance0, distance1, label in test_list:
                distance0 = distance0.unsqueeze(0).unsqueeze(0)
                distance1 = distance1.unsqueeze(0).unsqueeze(0)

                output1, output2 = siamese_net(distance0, distance1)
                euclidean_distance = F.pairwise_distance(output1, output2)
                if euclidean_distance > 2:
                    euclidean_distance = 2
                p_distanc.append(euclidean_distance)
                if euclidean_distance > 0.5:
                    # 认为不相似
                    label_predict = 1
                else:
                    label_predict = 0

                score_labe.append(label)
                score_labe_predict.append(label_predict)

            # print("p_distance:", p_distanc)
            # print("label:", score_labe)
            # print("label_predict:", score_labe_predict)
            print("accuracy_test:", accuracy_score(score_labe, score_labe_predict))

            similarity_test.append(p_distanc)

    display_image(similarity_train, number_train)
    display_image(similarity_test, number_test)


# 显示图象
def display_image(image, number):
    # 设置横轴，为训练时刻数
    x = []
    for i in range(200):
        x.append(i)

    plt.figure(figsize=(20, 15))

    # 设置纵轴，为相似度变化
    for i in range(number):
        y = [k[i] for k in image]
        plt.plot(x, y, '-o', label=str(i), linewidth=1.0)

    plt.legend()
    plt.show()
