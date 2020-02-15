import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import math
# 将变量初始化：梯度下降的初始点
x = tf.Variable(15.0, dtype=tf.float32)
# 函数
y = tf.pow(x-1, 2.0)
# 梯度下降，学习率设置为0.05
opti = tf.train.GradientDescentOptimizer(0.05).minimize(y)
# 画曲线
value = np.arange(-15, 17, 0.01)
y_value = np.power(value-1, 2.0)
plt.plot(value, y_value)
# 创建会话
session = tf.Session()
session.run(tf.global_variables_initializer())
# 150次迭代
for i in range(150):
    session.run(opti)
    if(i%10==9):
        v = session.run(x)
        plt.plot(v, math.pow(v-1, 2.0), 'go')
        print("第%d次的x的迭代值：%f" % (i+1, v))
plt.show()



