##
# date：2020/02/28
# author：zym
# description:implement of k_means++
##

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 生成samples：三堆点
X, category = datasets.make_blobs(n_samples=500, random_state=2)


# print(category[:10])


# 计算两向量之间的欧式距离
def calcuDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


# 生成k个中心点
def init_cluster_centers(X, k):
    # 中心点列表
    cluster_center = []

    # 随机选取第一个点
    random_id = np.random.choice((X.shape[0]))
    cluster_center.append(X[random_id])
    print(random_id)

    # 依据概率选择剩余中心点
    delta_x = (X[:, 0] - cluster_center[0][0])
    delta_y = (X[:, 1] - cluster_center[0][1])
    Dx2 = np.power(delta_x, 2) + np.power(delta_y, 2)
    sum_Dx2 = np.sum(Dx2)
    Px = Dx2 / sum_Dx2
    # 轮盘法选中心点
    rand_num = np.random.random()
    start = 0
    for ki in range(1, k):  # 除初始点以外还需要生成的中心点个数
        for index, pi in enumerate(Px):  # 遍历所有的概率值
            start += pi
            if rand_num <= start:
                cluster_center.append(X[index])
                break

    clus_cen = np.array(cluster_center)
    return clus_cen


# 计算样本点到聚类中心的最短距离，并将样本归类
def divide_category(X, clus_cen, k):
    # 用字典存储划分的k类数据，目的是同时存储category_id和对应的样本
    clusterDict = {}
    # 计算每个样本到k个聚类中心的距离
    for sample in X:  # 遍历样本
        vec1 = sample
        min = np.inf
        for i in range(k):  # 遍历k个聚类中心
            vec2 = clus_cen[i]
            distance = calcuDistance(vec1, vec2)
            if distance < min:
                min = distance
                category_temp = i
        if category_temp not in clusterDict.keys():
            clusterDict.setdefault(category_temp, [])
        clusterDict[category_temp].append(sample)

    return clusterDict


# 更新样聚类中心的坐标
def update_cluscen(clusterDict):
    new_center = []
    for key in clusterDict.keys():
        center_vec = np.mean(clusterDict[key], axis=0)
        new_center.append(center_vec)
    return new_center


# 计算相邻两次更新，中心点之间的距离和
def center_between_distance(center1, center2):
    sumdis = 0
    for i in range(len(center1)):
        vec1 = center1[i]
        vec2 = center2[i]
        dis = calcuDistance(vec1, vec2)
        sumdis += dis
    return sumdis


# 迭代更新
def converge(X, clus_cen, k):
    # 动态显示打开
    plt.ion()
    fig, ax = plt.subplots()

    i = 0
    sum_dis = np.inf
    centerDict = {'before': clus_cen}
    new_center = clus_cen
    while sum_dis > 0.00000000001:
        # 显示样本点和当前的聚类中心点
        plt.scatter(X[:, 0], X[:, 1], c=category)
        plt.scatter(np.array(new_center)[:, 0], np.array(new_center)[:, 1], c='red')

        i += 1
        clusterDict = divide_category(X, new_center, k)
        new_center = update_cluscen(clusterDict)

        # 显示部分
        plt.title('iteration count:{}'.format(i))
        plt.pause(0.5)
        ax.cla()

        if 'after' not in centerDict.keys():
            centerDict.setdefault('after', new_center)
        centerDict['after'] = new_center
        sum_dis = center_between_distance(centerDict['before'], centerDict['after'])
        centerDict['before'] = new_center
    plt.ioff()
    return new_center


clus_cen = init_cluster_centers(X, 3)
new_center = converge(X, clus_cen, 3)
# 显示最终结果
plt.scatter(X[:, 0], X[:, 1], c=category)
plt.scatter(np.array(new_center)[:, 0], np.array(new_center)[:, 1], c='red')
plt.title('iteration over!!')
plt.show()
