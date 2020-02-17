import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# 6个点的横坐标
x = tf.constant([1, 2, 3, 4, 5, 6], tf.float32)
# 6个点的纵坐标
y = tf.constant([3, 4, 7, 8, 11, 14], tf.float32)
# 初始化直线的斜率和截距
w = tf.Variable(1.0, dtype=tf.float32)
b = tf.Variable(1.0, dtype=tf.float32)
# 损失函数，即6个点到直线沿y轴方向距离的平方和
loss = tf.reduce_sum(tf.square(y-(w*x+b)))
# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 对损失函数利用梯度下降法计算w和b
opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
# 记录每次迭代后的平均误差（Mean Squared Error）
MSE = []
# 循环500次
for i in range(500):
    sess.run(opti)
    MSE.append(sess.run(loss))
    # 每隔50词打印直线的斜率和截距
    if i%50==9:
        print((sess.run(w), sess.run(b)))
# 画出损失函数的变化曲线图
plt.figure(1)
plt.plot(MSE)
plt.show()
# 画出6个点和最后计算得到的直线
plt.figure(2)
x_array, y_array = sess.run([x, y])
plt.plot(x_array, y_array, 'o')
xx = np.arange(0, 10, 0.05)
yy = sess.run(w)*xx+sess.run(b)
plt.plot(xx, yy)
plt.show()