import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# x的取值
x_value = np.arange(-10, 10, 0.01)
sess = tf.Session()
# x为自变量通过sigmoid函数后得到y_value
y_value = sess.run(tf.nn.sigmoid(x_value))
# 对sigmoid求导f'(x) = f(x)*(1-f(x))
Y_value = y_value*(1-y_value)
# 画sigmoid图
plt.figure(1)
plt.plot(x_value, y_value)
plt.show()
# 画sigmoid导数的图
plt.figure(2)
plt.plot(x_value, Y_value)
plt.show()
