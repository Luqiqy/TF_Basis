import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# xyz坐标内的固定点
xy = tf.placeholder(tf.float32, [None, 2])
z = tf.placeholder(tf.float32, [None, 1])
# 初始化z=w1*x+w2*y+b中的w1，w2和b
w = tf.Variable(tf.constant([[1], [1]], tf.float32), dtype=tf.float32, name='w')
b = tf.Variable(1.0, dtype=tf.float32, name='b')
# 损失函数
loss = tf.reduce_sum(tf.square(z-(tf.matmul(xy, w)+b)))
# 创建会话
sess = tf.Session()
# variable初始化
sess.run(tf.global_variables_initializer())
# 梯度下降法
opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
# 用来记录每一次迭代后的均方误差
MSE = []
# 训练数据
xy_train = np.array([[1, 1],
                     [2, 1],
                     [3, 2],
                     [1, 2],
                     [4, 5],
                     [5, 8]], np.float32)
z_train = np.array([[8], [12], [10], [14], [28], [10]], np.float32)
# 声明一个tf.train.Saver类
saver = tf.train.Saver()
# 训练模型，迭代500次
for i in range(500):
    # 梯度下降法
    sess.run(opti, feed_dict={xy: xy_train, z: z_train})
    # 计算每一次迭代后的损失值并且添加到列表保存
    MSE.append(sess.run(loss, feed_dict={xy: xy_train, z: z_train}))
    # 每隔100次打印w和b
    if i % 100 == 0:
        saver.save(sess, './L3model/model.ckpt', global_step=i)
        print("------第"+str(i)+"次迭代值------")
        print(sess.run([w, b]))
# 打印最后一次迭代值
print("------第"+str(500)+"次迭代值------")
print(sess.run([w, b]))
saver.save(sess, './L3model/model.ckpt', global_step=i)
# 画损失函数的图像
plt.figure(1)
plt.plot(MSE)
plt.show()