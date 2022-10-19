#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :adv
# @Time      :2022/5/4 21:26
# @Author    :Ouyang Bin
import torch
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

from FL_base_function import tst
from AdvBox.adversarialbox.adversary import Adversary
from AdvBox.adversarialbox.attacks.deepfool import DeepFoolAttack
from AdvBox.adversarialbox.models.pytorch import PytorchModel


# distance.shape = 40x11x20
def inference_attack(all_GMs, train_loader, test_loader, device, FL_params):
    train_input = []
    test_input = []
    # 取多组数据
    for each_attack_epoch in range(FL_params.attack_epoch):  # attack_epoch=40
        # 生成一个0-64随机数
        data = random.randint(0, 63)
        # 时间轴数组，存放每个数据随着训练轮数变化，距离值的变化
        # 最后为
        Distance_train = []
        Distance_test = []
        # 对联邦学习过程中的每次全局模型进行分析
        for each_epoch in range(FL_params.global_epoch + 1):  # global_epoch=20
            # 目标模型
            model = all_GMs[each_epoch]

            # 打印选取目标模型在测试集上的准确率
            print("attack_epoch: {}, round: {}".format(each_attack_epoch + 1, each_epoch))
            if not each_attack_epoch:
                tst(model, test_loader)

            model = model.to(device)
            model = model.eval()

            # 设置为不保存梯度值 自然也无法修改
            for param in model.parameters():
                param.requires_grad = False

            input_min = test_loader.sampler.data_source.tensors[0].min()
            input_max = test_loader.sampler.data_source.tensors[0].max()
            bounds = (input_min, input_max)
            M_tgt = PytorchModel(model, None, bounds, channel_axis=1, nb_classes=2)

            deepfool_attacker = DeepFoolAttack(M_tgt)
            attacker_config = {"iterations": 500, "overshoot": 0.02}

            print("train_predict...")

            # distance_train存放的是每一轮中各个数据点的距离值
            distance_train = []
            # 进入主循环(训练集)
            for _, (XX_tgt, YY_tgt) in enumerate(train_loader):

                # XX_tgt.shape = [64, 108] -> [1, 108]
                # 在一个epoch(64个)中随机取一个数据点
                XX_tgt = XX_tgt[data, :]
                XX_tgt = XX_tgt.unsqueeze(0)
                XX_tgt = XX_tgt.cpu().numpy()

                # 生成一个扰动样本作为测试样本
                XX_tgt_new_1 = copy.deepcopy(XX_tgt)
                XX_tgt_new_1[0, 102] += 0.2
                XX_tgt_new_2 = copy.deepcopy(XX_tgt)
                XX_tgt_new_2[0, 0:9] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_3 = copy.deepcopy(XX_tgt)
                XX_tgt_new_3[0, 103] += 0.1
                XX_tgt_new_4 = copy.deepcopy(XX_tgt)
                XX_tgt_new_4[0, 9:25] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_5 = copy.deepcopy(XX_tgt)
                XX_tgt_new_5[0, 104] += 0.5
                XX_tgt_new_6 = copy.deepcopy(XX_tgt)
                XX_tgt_new_6[0, 25:32] = [1, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_7 = copy.deepcopy(XX_tgt)
                XX_tgt_new_7[0, 32:47] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_8 = copy.deepcopy(XX_tgt)
                XX_tgt_new_8[0, 47:53] = [0, 1, 0, 0, 0, 0]
                XX_tgt_new_9 = copy.deepcopy(XX_tgt)
                XX_tgt_new_9[0, 53:58] = [1, 0, 0, 0, 0]
                XX_tgt_new_10 = copy.deepcopy(XX_tgt)
                XX_tgt_new_10[0, 58:60] = [1, 0]
                list_label = [XX_tgt, XX_tgt_new_1, XX_tgt_new_2, XX_tgt_new_3,
                              XX_tgt_new_4, XX_tgt_new_5, XX_tgt_new_6, XX_tgt_new_7,
                              XX_tgt_new_8, XX_tgt_new_9, XX_tgt_new_10]

                YY_tgt = None

                for each_XX_tgt in list_label:
                    adversary = Adversary(each_XX_tgt, YY_tgt)
                    adversary = deepfool_attacker(adversary, **attacker_config)

                    if adversary.is_successful():
                        advs = adversary.adversarial_example[0]

                        # 对抗成功的最小的扰动值
                        d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))

                        print("attack success, adv_label={}, distance={}".format(adversary.adversarial_label, d))

                    else:
                        advs = adversary.bad_adversarial_example[0]

                        d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))

                        print("attack failed, distance=", d)

                    distance_train.append(d)

                # 只进行一轮
                break

            print("test_predict...")

            distance_test = []
            # 进入主循环(测试集)
            for _, (XX_tgt, YY_tgt) in enumerate(test_loader):

                # XX_tgt.shape = [64, 108] -> [1, 108]
                # 在一批数据(64)中随机取一个数据点
                XX_tgt = XX_tgt[data, :]
                XX_tgt = XX_tgt.unsqueeze(0)
                XX_tgt = XX_tgt.cpu().numpy()

                # 生成一个扰动样本作为测试样本
                XX_tgt_new_1 = copy.deepcopy(XX_tgt)
                XX_tgt_new_1[0, 102] += 0.2
                XX_tgt_new_2 = copy.deepcopy(XX_tgt)
                XX_tgt_new_2[0, 0:9] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_3 = copy.deepcopy(XX_tgt)
                XX_tgt_new_3[0, 103] += 0.1
                XX_tgt_new_4 = copy.deepcopy(XX_tgt)
                XX_tgt_new_4[0, 9:25] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_5 = copy.deepcopy(XX_tgt)
                XX_tgt_new_5[0, 104] += 0.5
                XX_tgt_new_6 = copy.deepcopy(XX_tgt)
                XX_tgt_new_6[0, 25:32] = [1, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_7 = copy.deepcopy(XX_tgt)
                XX_tgt_new_7[0, 32:47] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_8 = copy.deepcopy(XX_tgt)
                XX_tgt_new_8[0, 47:53] = [0, 1, 0, 0, 0, 0]
                XX_tgt_new_9 = copy.deepcopy(XX_tgt)
                XX_tgt_new_9[0, 53:58] = [1, 0, 0, 0, 0]
                XX_tgt_new_10 = copy.deepcopy(XX_tgt)
                XX_tgt_new_10[0, 58:60] = [1, 0]
                list_label = [XX_tgt, XX_tgt_new_1, XX_tgt_new_2, XX_tgt_new_3,
                              XX_tgt_new_4, XX_tgt_new_5, XX_tgt_new_6, XX_tgt_new_7,
                              XX_tgt_new_8, XX_tgt_new_9, XX_tgt_new_10]

                # list_label = feature_extract(XX_tgt, FL_params.n_feature)

                YY_tgt = None

                for each_XX_tgt in list_label:

                    adversary = Adversary(each_XX_tgt, YY_tgt)
                    adversary = deepfool_attacker(adversary, **attacker_config)

                    if adversary.is_successful():
                        advs = adversary.adversarial_example[0]

                        # 对抗成功的最小的扰动值
                        d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))
                        print("attack success, adv_label={}, distance={}".format(adversary.adversarial_label, d))

                    else:
                        advs = adversary.bad_adversarial_example[0]
                        d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))
                        print("attack failed, distance=", d)

                    distance_test.append(d)

                # 只进行一轮
                break

            Distance_train.append(distance_train)
            Distance_test.append(distance_test)

        # 进行转置，便于观察 21x11 -> 11x21
        Distance_train = [[row[i] for row in Distance_train] for i in range(len(Distance_train[0]))]
        Distance_test = [[row[i] for row in Distance_test] for i in range(len(Distance_test[0]))]

        # display_image(Distance_train, "train")
        # display_image(Distance_test, "test")

        train_input.append(Distance_train)  # 40x11x21
        test_input.append(Distance_test)

    train_input = torch.tensor(train_input)
    test_input = torch.tensor(test_input)

    torch.set_printoptions(threshold=np.inf)  # 数据个数超过np.inf无穷后折叠

    # print("train_input:", train_input)
    # print("test_input:", test_input)

    return train_input, test_input


# 测试训练模型精度专用，得到一个batch
def for_test(all_GMs, train_loader, test_loader, device, FL_params):
    # 目标模型
    for i in range(FL_params.global_epoch):
        model = all_GMs[i]

        # 打印选取目标模型在测试集上的准确率
        tst(model, train_loader)

        print("**********************************")

        tst(model, test_loader)


# 显示图象
def display_image15(image, mode):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    plt.figure(figsize=(20, 15))

    for i in range(len(image)):
        plt.plot(x, image[i], '-o', label=mode + " property " + str(i), linewidth=2.0)

    plt.rcParams.update({'font.size': 5})
    plt.legend(loc='upper right')  # 显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()


# 显示图象
def display_image3(image, mode):
    x = [0, 1, 2]

    plt.figure(figsize=(20, 15))

    for i in range(len(image)):
        plt.plot(x, image[i], '-o', label=mode + " property " + str(i), linewidth=2.0)

    plt.rcParams.update({'font.size': 5})
    plt.legend(loc='upper right')  # 显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()
