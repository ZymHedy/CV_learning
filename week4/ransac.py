#  Ransac many noisy data

import numpy as np
import matplotlib.pyplot as plt
import random

SIZE = 50
ERROR = 100

x = np.linspace(0, 10, SIZE)
y = 3 * x + 10
random_x = [x[i] + random.uniform(-0.5, 0.5) for i in range(SIZE)]
random_y = [y[i] + random.uniform(-0.5, 0.5) for i in range(SIZE)]

for i in range(ERROR):
    random_x.append(random.uniform(0, 20))
    random_y.append(random.uniform(10, 40))

RANDOM_X = np.array(random_x)
RANDOM_Y = np.array(random_y)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(RANDOM_X, RANDOM_Y)
plt.show()

# OLS
from sklearn.linear_model import LinearRegression

reg = LinearRegression(fit_intercept=True)
reg.fit(RANDOM_X.reshape(-1, 1), RANDOM_Y.reshape(-1, 1))
slope = reg.coef_
intercept = reg.intercept_
PREDICT_Y = reg.predict(RANDOM_X.reshape(-1, 1))
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(RANDOM_X, RANDOM_Y)
ax1.plot(RANDOM_X, PREDICT_Y, c='red')
plt.show()

## RANSAC
iterations = 100
tolerent_sigma = 1
thresh_size = 0.5

best_slope = -1
best_intercept = 0
pretotal = 0

plt.ion()
plt.figure()
for i in range(iterations):
    sample_index = random.sample(range(SIZE + ERROR), 2)
    x_1 = RANDOM_X[sample_index[0]]
    x_2 = RANDOM_X[sample_index[1]]
    y_1 = RANDOM_Y[sample_index[0]]
    y_2 = RANDOM_Y[sample_index[1]]

    # y = ax +b
    slope = (y_2 - y_1) / (x_2 - x_1)
    intercept = y_1 - slope * x_1

    # calculate inliers
    total_inliers = 0
    for index in range(SIZE + ERROR):
        PREDICT_Y = slope * RANDOM_X[index] + intercept
        if abs(PREDICT_Y - RANDOM_Y[index]) < tolerent_sigma:
            total_inliers += 1

    if total_inliers > pretotal:
        pretotal = total_inliers
        best_slope = slope
        best_intercept = intercept

    if total_inliers > (SIZE + ERROR) * thresh_size:
        break

    plt.title(f"RANSAC in Linear Regression: Iter {i + 1}, Inliers {pretotal}")
    plt.scatter(RANDOM_X, RANDOM_Y)
    Y = best_slope * RANDOM_X + best_intercept
    plt.plot(RANDOM_X, Y,'black')
    plt.pause(1)
    plt.clf()
