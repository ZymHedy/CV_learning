##
# date:2020/02/12
# Author:Zym
# Description:ransac about linear model
#
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression

SIZE = 50  # 内点数
ERROR = 100  # 完全噪音点数量(外点)


def gen_data(size, error):
    # 完全没有噪音的数据
    x = np.linspace(0, 10, SIZE)
    y = 3 * x + 10
    random_x = [x[i] + random.uniform(-0.5, 0.5) for i in range(SIZE)]  # uniform是均匀分布
    random_y = [y[i] + random.uniform(-0.5, 0.5) for i in range(SIZE)]

    # 为数据添加噪音
    for i in range(ERROR):
        random_x.append(random.uniform(0, 20))
        random_y.append(random.uniform(10, 40))

    random_x = np.array(random_x)
    random_y = np.array(random_y)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # ax1.scatter(random_x, random_y)
    # plt.show()
    return random_x, random_y


# x,y是numpy类型的一维列表
x, y = gen_data(SIZE, ERROR)

# 测试最小二乘法拟合直线：线性回归
reg = LinearRegression(fit_intercept=True)  # 需要截距
reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # 参数是二维矩阵类型的,完成训练
slope = reg.coef_  # 斜率
intercept = reg.intercept_  # 截距
PREDICT_Y = reg.predict(x.reshape(-1, 1))
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(x, y)
ax1.plot(x, PREDICT_Y, c='red')
plt.show()

# 测试发现最小二乘法会考虑到所有很夸张的噪点，但是我们只希望找到实际的符合需求的模型
# 因此需要ransac

# ransac
iterations = 100
tolerant_sigma = 1  # 预测点和实际点之间的偏差
thresh_size = 0.5  # 目标的内点数百分比

# 应该从样本集中随机取两个点进行斜率和截距的初始化
# 但是此处人为设定了值
best_slope = -1
best_intercept = 0
pre_total = 0

plt.ion()
plt.figure()

for i in range(iterations):
    sample_index = random.sample(range(SIZE + ERROR), 2)
    x_1 = x[sample_index[0]]
    x_2 = x[sample_index[1]]
    y_1 = y[sample_index[0]]
    y_2 = y[sample_index[1]]

    # y = ax + b
    slope = (y_2 - y_1) / (x_2 - x_1)
    intercept = y_1 - slope * x_1

    # calculate inliers计算内点个数
    total_inliers = 0
    for index in range(SIZE + ERROR):
        PREDICT_Y = slope * x[index] + intercept
        if abs(PREDICT_Y - y[index]) < tolerant_sigma:
            total_inliers += 1

    # 如果本次迭代表现更优，则保存相关参数
    if total_inliers > pre_total:
        pre_total = total_inliers
        best_slope = slope
        best_intercept = intercept

    # 终止条件，内点数达到目标比例
    if total_inliers > (SIZE + ERROR) * thresh_size:
        break

    best_Y = best_slope * x + best_intercept

    plt.title('ransac in linear regression:iter{} inliers:{}'.format(i + 1, pre_total))
    plt.scatter(x, y)
    plt.plot(x, best_Y, c='black')
    plt.pause(1)
    plt.clf()
