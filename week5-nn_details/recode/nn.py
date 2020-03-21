import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 初始化
np.random.seed(666)  # 生成随机种子
N = 200
nn_input_dim = 2
nn_output_dim = 2  # one_hot encoding
lr = 0.1
# reg_lamda = 0.01

# 选择sklearn的数据集
X, y = datasets.make_moons(n_samples=N, noise=0.1)
print(len(y))


# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# caculate loss
def caculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1  # shape:(200,20),引入hidden_layer
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2  # shape:(200,2)

    # softmax
    exp_scores = np.exp(z2)  # 对z2的每个元素都做e的幂次方，因此shape仍为(200,2)
    # np.sum.shape:(200,1),axis=1表示对每行中的元素进行相加，keepdims维持shape为(200,1),否则为(200,)
    # probs.shape：(200,2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # cross_entropy
    # 此处相当于自动把y标签变成one_hot编码
    # 标签为0，则one_hot=[1,0];标签为1，则one_hot=[0,1]
    # 故直接第0类取对应的第一个分量，第1类取对应的第二个分量
    # ce = -sum(yi*log(probs))
    log_probs = -np.log(probs[range(N), y])
    sum_loss = np.sum(log_probs)
    return sum_loss / N


# 建立模型
def build_model(nn_hidden_dim, num_iteration=10000, print_loss=True):
    W1 = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hidden_dim))
    W2 = np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)
    b2 = np.zeros((1, nn_output_dim))
    model = {}  # 模型定义为字典类型

    # 参数更新
    # step1:正向传递
    # step2:反向传播
    for i in range(num_iteration):

        # 正向传递
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        # softmax的输出：probs.shape[200,2],一组概率值
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # 记录一下增量
        delta3 = probs

        # 反向传播:利用梯度下降更新每一层的权值和偏置，即更新W1,b1,W2,b2
        # dW2=(偏loss/偏z2)*(偏z2/偏W2)--chain_rule
        # db2=sum()
        # dW1=(偏a1/偏z1)*(偏z1/偏W1)
        # db1=
        delta3[range(N), y] -= 1
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # 梯度下降回传
        W1 -= lr * dW1
        W2 -= lr * dW2
        b1 -= lr * db1
        b2 -= lr * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print('loss after iteration {}:{}'.format(i, caculate_loss(model)))

    return model


model = build_model(nn_hidden_dim=20, print_loss=True)
