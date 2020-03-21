##
# data: 2020/2/11
# Author: Zym
# Description: endless recode of linear regression
##
import numpy as np
import matplotlib.pyplot as plt
import random


# 生成数据
# 预测y值
# 计算loss：均方差
# 梯度下降

def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()
    print('w:{}'.format(w))
    sampleNo = 100
    x_list = []
    y_list = []
    for i in range(sampleNo):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 100)
        # print('第：{}次的y：{}'.format(i, y))
        x_list.append(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


def predict(x, w, b):
    pred_y = w * x + b
    return pred_y


def loss(gt_x, gt_y, w, b):
    pred_y = predict(gt_x, w, b)
    diff = pred_y - gt_y
    loss = np.sum(diff ** 2) / 2 * len(diff)
    return loss


def gradient(gt_x, gt_y, w, b, lr):
    pred_y = predict(gt_x, w, b)
    diff = pred_y - gt_y
    dw = np.sum(diff * gt_x)/len(diff)
    db = np.sum(diff)/len(diff)
    w -= lr * dw
    b -= lr * db
    return w, b


def train(gt_x, gt_y, lr, max_iter):
    plt.ion()
    fig, ax = plt.subplots()
    w = 1
    b = 0
    for i in range(max_iter):
        w,b = gradient(gt_x,gt_y,w,b,lr)
        pred_y = predict(gt_x,w,b)

        plt.scatter(gt_x, gt_y)
        plt.plot(gt_x,pred_y)
        plt.pause(0.5)
        if i != max_iter-1:
            ax.cla()

        print('w update:{}',format(w))
        print('loss:{}'.format(loss(gt_x,gt_y,w,b)))

    plt.ioff()




x, y = gen_sample_data()
# # print(y)
# plt.figure()
# plt.scatter(x, y)
# plt.show()

train(x,y,0.0003,100)
