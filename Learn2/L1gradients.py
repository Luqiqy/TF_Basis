import tensorflow.compat.v1 as tf
import numpy as np
# 通过矩阵乘法构建函数F=(3x1+4x2)**2
w = tf.constant([[3, 4]], tf.float32)
x = tf.placeholder(tf.float32, (2, 1))
y = tf.matmul(w, x)
F = tf.pow(y, 2)
# 开始计算函数F的梯度
grads = tf.gradients(F, x)
session = tf.Session()
# 打印结果
print(session.run(grads, {x: np.array([[2], [3]])}))

