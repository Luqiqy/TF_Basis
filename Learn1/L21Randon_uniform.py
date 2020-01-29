import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
# 构造四维张量，即10个深度为3的4行5列的三维张量，每个值在区间[0,10]满足均匀分布
x = tf.random_uniform([10, 3, 4, 5], minval=0, maxval=10, dtype=tf.float32)
# 查看该四维张量的尺寸
print(x.get_shape())
# 将该Tensor转换为nddarry
session = tf.Session()
array = x.eval(session=session)
# 查看array的shape
print(np.shape(array))
print(type(array), tf.shape(array))
# 将array转化为1个一维的ndarray来画图
array1d=array.reshape([-1])
plt.hist(array1d)
plt.show()