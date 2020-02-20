import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# x的取值
x_value = np.arange(-10, 10, 0.01)
sess = tf.Session()
# x为自变量通过tanh函数后得到y_value
y_value = sess.run(tf.nn.tanh(x_value))
# 对tanh求导
Y_value = 1-np.power(y_value, 2.0)
# 画tanh图
plt.figure(1)
plt.plot(x_value, y_value)
plt.show()
# 画tanh导数的图
plt.figure(2)
plt.plot(x_value, Y_value)
plt.show()
