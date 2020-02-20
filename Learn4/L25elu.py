import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# x的取值
x_value = np.arange(-10, 10, 0.01)
sess = tf.Session()
# x为自变量通过elu函数后得到y_value
y_value = sess.run(tf.nn.elu(x_value))
# 画elu图
plt.figure(1)
plt.plot(x_value, y_value)
plt.show()
