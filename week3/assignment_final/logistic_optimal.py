##
# Data: 2020/2/10 Mon
# Author: Yimeng Zhang
# Description: dynamic showing of logistic
##
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt


# 假定二维变量的二分类问题
# make dataset
# ground truth label: 0 or 1
# predict probs: (0,1)
# logistic loss

# 生成两簇数据x_posi,x_zero，每簇1000个样本，每个样本两个属性x1,x2
def gen_sample_data():
    sampleNo = 1000
    miu = np.array([1, 5])
    matrix = np.array([[2, 0], [0, 3]])
    R = cholesky(matrix)
    s = np.dot(np.random.randn(sampleNo, 2), R) + miu
    x_posi = np.hstack((s, np.ones((sampleNo, 1))))
    # plt.plot(s[:, 0], s[:, 1], "+")

    miu = np.array([6, 0])
    matrix = np.array([[2, 1], [1, 2]])
    R = cholesky(matrix)
    s = np.dot(np.random.randn(sampleNo, 2), R) + miu
    x_zero = np.hstack((s, np.zeros((sampleNo, 1))))
    # plt.plot(s[:, 0], s[:, 1], "x")
    # plt.show()

    X = np.vstack((x_posi, x_zero))
    return X


class logistic():
    def __init__(self):
        self.w1 = 0
        self.w2 = 0
        self.b = 0

    # inference from data to probs
    def sigmoid(self, x):
        pred_y = 1 / (1 + np.exp(-(self.w1 * x[:, 0] + self.w2 * x[:, 1] + self.b)))
        return pred_y

    # cost function
    def eval_loss(self, x, y):
        loss = -(y * np.log(self.sigmoid(x)) + (1 - y) * np.log(1 - self.sigmoid(x)))
        return np.mean(loss)

    # single sample's gradient
    def gradient(self, x, y, pred_y):
        diff = pred_y - y
        dw1 = diff * x[:, 0]
        dw2 = diff * x[:, 1]
        db = diff
        return dw1, dw2, db

    # update w1,w2,b
    # 随机梯度下降，每次只取一部分样本，不取全局的
    def cal_step_gradient(self, batch_x, batch_y, lr):
        pred_y = self.sigmoid(batch_x)
        dw1, dw2, db = self.gradient(batch_x, batch_y, pred_y)
        self.w1 -= lr * np.mean(dw1)
        self.w2 -= lr * np.mean(dw2)
        self.b -= lr * np.mean(db)

    # 加上plt动态显示的训练过程
    def train(self, x, batch_size, lr, max_iter):
        # 增加plt的动态显示
        x_axe = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 1000)  # 把从最小值到最大值中间的区间等分成1000段
        plt.ion()  # 打开plt的互动实时操作
        fig, ax = plt.subplots()  # ax是fig画布上的一块图像

        # 迭代的过程
        for i in range(max_iter):
            # 随机取出标号
            batch_idx = np.random.choice(len(x), batch_size, False)
            batch_x = np.array([x[j][:2] for j in batch_idx])
            batch_y = np.array([x[j][2] for j in batch_idx])
            self.cal_step_gradient(batch_x, batch_y, lr)
            print('w1:{} w2:{} b:{}'.format(self.w1, self.w2, self.b))
            print('loss:{}'.format(self.eval_loss(batch_x, batch_y)))

            plt.xlim(np.min(x[:, 0]) * 1.1, np.max(x[:, 0]) * 1.1)  # 设置坐标轴的xy范围
            plt.ylim(np.min(x[:, 1]) * 1.1, np.max(x[:, 1]) * 1.1)
            plt.scatter(x[:, 0], x[:, 1], c=x[:, 2])  # 生成散点图，且c依据标签值显示颜色
            # x_axe*w1+y_axe*w2+b=0 决策边界，固定x1，x2应该由不断迭代的w1，w2,b来决定
            # construct line with predict params w1,w2,b
            y_axe = (-self.b - x_axe * self.w1) / self.w2
            plt.plot(x_axe, y_axe,linewidth=2)
            plt.title('logistic_regression iter:{}'.format(i+1))
            plt.pause(0.5)  # 为了人眼观看暂停
            if i != max_iter - 1:
                ax.cla()  # 清除当前画布上的图像

        return self.w1, self.w2, self.b


X = gen_sample_data()
lgr = logistic()
lgr.train(X, 100, 0.1, 50)
