from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # fit4组数据3个状态
code = enc.transform([[0, 1, 3]]).toarray()  # 分别对4组数据三个状态进行编码
print(code)
