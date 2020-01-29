import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import math
# 正态分布标准差sigma=1，均值mu=10
sigma = 1
mu = 10
# 构造四位张量，即10个深度为3行为4列为5的三维向量，张量中的值满足均值为10，标准差为1的正态分布
result = tf.random_normal([10, 3, 4, 5], mu, sigma, tf.float32)
# 将Tensor转化为ndarray
session = tf.Session()
array = session.run(result)
# 将array转换为1个一维的ndarray
array1d = array.reshape([-1])
# 计算并显示直方图
histogram, bins, patch = plt.hist(array1d, 25, facecolor='gray', alpha=0.5, normed=True)
x = np.arange(5, 15, 0.01)
y = 1.0/(math.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x-mu, 2.0)/(2*math.pow(sigma,2)))
plt.plot(x, y)
plt.show()
