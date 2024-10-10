import matplotlib.pyplot as plt
import numpy as np


def partition_matrix(X, center, m):
    out = np.zeros((center.shape[0], X.shape[0]))
    if center.shape[1] > 1:
        for k in range(center.shape[0]):
            out[k, :] = np.sqrt(
                np.sum(((X - np.ones((X.shape[0], 1)) * center[k, :])) ** 2, axis=1)
            )
    else:  # 1-D data
        for k in range(center.shape[0]):
            out[k, :] = np.abs(center[k] - X).T
    tmp = out ** (-2 / (m - 1))
    y = tmp / (np.ones((center.shape[0], 1)) * np.sum(tmp, axis=0))
    return y


def fcm(X, c=3, m=2, max_iter=100, epsilon=1e-5):
    """
    FCM algorithm. Numpy verison.

    Args:
        X (_type_): 输入
        c (int, optional): 聚类个数. Defaults to 3.
        m (int, optional): 模糊指数. Defaults to 2.
        max_iter (int, optional): 最大迭代次数. Defaults to 100.
        epsilon (_type_, optional): 收敛标准. Defaults to 1e-5.
    """
    n = X.shape[0]
    centers = np.random.rand(c, X.shape[1])
    u = np.random.rand(n, c)
    u = u / np.sum(u, axis=1, keepdims=1)
    for i in range(max_iter):
        # 更新模糊聚类中心
        for j in range(c):
            centers[j, :] = np.sum((u[:, j] ** m).reshape(-1, 1) * X, axis=0) / np.sum(
                u[:, j] ** m
            )

        # 更新模糊指数 u
        dist = np.sum((X.reshape(n, 1, X.shape[1]) - centers) ** 2, axis=2)
        u_new = 1 / (dist ** (1 / (m - 1)))
        u_new = u_new / np.sum(u_new, axis=1).reshape(-1, 1)

        # 判断停止条件
        if np.max(np.abs(u_new - u)) < epsilon:
            break
        else:
            u = u_new.copy()
    return centers, u


def plot(x, y, plttype="scatter"):
    assert plttype in ["scatter", "plot"]

    if type == "plot":
        sort_idx = x.argsort()
        x = x[sort_idx]
        y = y[sort_idx]
        plt.plot(x, y)
    else:
        plt.scatter(x, y)
    plt.title("Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
