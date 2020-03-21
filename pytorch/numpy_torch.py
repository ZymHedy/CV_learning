import torch
import numpy as np

# numpy array和torch tensor的相互转换
np_data = np.arange(6).reshape(2, 3)  # arange不要搞错，是表示一个范围
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
# print(np_data)
# print(torch_data)
# print(tensor2array)

# abs绝对值
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
# print(np.mean(data))
# print(torch.mean(tensor))

# 矩阵运算
data2 = [[1, 2], [3, 4]]
tensor2 = torch.FloatTensor(data2)
print(np.matmul(data2, data2))  # matrix multiply
print(torch.mm(tensor2, tensor2))  # mm直接是上面的缩写
