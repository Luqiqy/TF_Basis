import tensorflow.compat.v1 as tf
import numpy as np
# 输入已知数据，用来训练，等号右为一矩阵，左为一矩阵
xy = tf.placeholder(tf.float32, [None, 2])
z = tf.placeholder(tf.float32, [None, 1])
# 初始化变量，矩阵形式
w = tf.Variable(tf.constant([[1], [1]], tf.float32), dtype=tf.float32, name='w')
# 构造损失函数
loss = tf.reduce_sum(tf.square(z-tf.matmul(xy, w)))
# 使用梯度下降法求解变量
opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
# 待训练的数据，等号右为一矩阵，左为一矩阵
xy_train = np.array([[1, 1], [2, 1], [3, 2],
                     [1, 2], [4, 5], [5, 8]], np.float32)
z_train=np.array([[8], [12], [10], [14], [28], [10]], np.float32)
# 创建会话，训练模型
session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(500):
    session.run(opti, feed_dict={xy: xy_train, z: z_train})
