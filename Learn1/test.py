'''
import tensorflow.compat.v1 as tf
import numpy as np
a = tf.constant([[[1],[2]],
                 [[2],[1]]])
print(a.get_shape())
session = tf.Session()
array = session.run(a)
print(array)
print(np.shape(array))
'''
import tensorflow as tf
# 构造三维张量
"""t = tf.constant([
                [[2, 5], [3, 3], [8, 2]],
                [[6, 1], [1, 2], [5, 6]],
                [[7, 9], [2, -3], [-1, 3]]
                ],
    tf.float32)
"""
"""
t = tf.constant([[[1, 0, 1],
                    [0, 1, 0]],
                   [[2, 0, 2],
                    [0, 2, 3]]], tf.float32)
print(t.get_shape())
# 从t的位置"[0, 0, 1]"开始，取范围为3深度x2行x2列的区域的值
# y = tf.slice(t, [1, 0, 1], [2, 2, 1])
# 创建会话
session = tf.Session()
print("原张量：\n", session.run(t))
# print("访问的张量：\n", session.run(y))
"""
