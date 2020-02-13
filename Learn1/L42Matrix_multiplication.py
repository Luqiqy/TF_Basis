import tensorflow.compat.v1 as tf
# 构建矩阵（二维张量）M x N
x = tf.constant([[1, 2], [3, 4]], tf.float32)
# 构建矩阵（二维张量）N x X
w = tf.constant([[-1], [-2]], tf.float32)
# 计算矩阵相乘
y = tf.matmul(x, w)
session = tf.Session()
# 打印结果
print("矩阵相乘结果：\n", session.run(y))