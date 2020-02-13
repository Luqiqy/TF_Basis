import tensorflow.compat.v1 as tf
import numpy as np
# 创建3行2列二维张量w
w = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]], tf.float32)
# 占位符
x = tf.placeholder(tf.float32, [2, None], name='x')
# 进行矩阵相乘
y = tf.matmul(w, x)
session = tf.Session()
# 给占位符x赋值为2行2列的二维向量，并得到结果
result1 = session.run(y, feed_dict={x:np.array([[2, 1], [1, 2]], np.float32)})
# 给占位符x赋值为2行1列的二维向量，并得到结果
result2 = session.run(y, feed_dict={x:np.array([[-1],
                                                [2]], np.float32)})
# 打印结果
print("x为2行2列：\n", result1)
print("x为2行1列：\n", result2)