# description:test for huge value computing in softmax
import numpy as np


# x是列向量
def softmax(x):
    # 将分子分母同除以e的最大指数
    x = x - np.max(x) 
    exps = np.exp(x)
    return np.exp(x) / np.sum(exps)


print(softmax(np.array([200000, 300000, 400000])))
