from math import ceil
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.io as io
from utils import plot, fcm, partition_matrix
from sklearn.preprocessing import MinMaxScaler
CLUSTER_CENTER_NUM = [7]  # number of cluster centers


TRAIN_DATA = np.load('radar.npz')['train_data'] #加载训练数据
TRAIN_INPUTS = TRAIN_DATA[:,:5]
TRAIN_TARGETS = TRAIN_DATA[:,5:]
TEST_DATA = np.load('radar.npz')['test_data'] #加载测试数据
#加载测试集Y
 
DIMS = 5
for cluster_center in CLUSTER_CENTER_NUM:
     # 采用FCM求解原型, fcm的函数自行完成
    centers, _ = fcm(TRAIN_DATA, c=cluster_center, m=2, max_iter=200, epsilon=1e-5)
    V = centers[:, :-1]
    print('V', V.shape)
    W = centers[:, -1:]
    print('W', W.shape)
     # 根据FCM中的公式求解输入空间的隶属度矩阵，partition_matrix函数根据公式自行完成
    Mem = partition_matrix(TRAIN_INPUTS, V, m=2)  
     
    trn_zz = []
    for ii in range(cluster_center):
         # 这一部分计算 Z，自行完成 ：trn_zz
         
    #trn_zz = np.concatenate(trn_zz, axis=1)
     
    trn_q = np.dot(Mem.T, W)
     
    trn_p = TRAIN_TARGETS - trn_q
     
    print("++++++++++++")
     # 利用最小二乘法求解向量 a
    a = # 采用最小二乘法计算a，自行完成
    print('a',a.shape)

    # 通过聚合表示输出
    trn_output = trn_q + np.dot(trn_zz, a)
    
    # 计算实际输出和预测输出的RMSE
    print(f"均方根误差(RMSE)：{np.sqrt(mean_squared_error(trn_output, TRAIN_TARGETS))}")

# 根据上述代码，基于训练得到的原型和参数向量a，实现测试集的预测，并输出 tst_output
# 这里没有提供TEST_TARGETS的结果，无需计算测试集的RMSE