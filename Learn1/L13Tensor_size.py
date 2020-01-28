# 利用Tensorflow中的函数shape得到张量的尺寸
'''
import tensorflow.compat.v1 as tf
t = tf.constant([[1, 2, 3],
                 [2, 0, 2],
                 [3, 2, 1]], tf.float32)
session = tf.Session()
s = tf.shape(t)
print("Tensor_size:", session.run(s))
'''
# 利用成员函数get_shape()或成员变量shape得到张量的尺寸
import tensorflow.compat.v1 as tf
t = tf.constant([[1, 2, 3],
                 [2, 0, 2],
                 [3, 2, 1]], tf.float32)
#s = t.shape
#print(s, type(s))

a = t.get_shape()
print(a, type(a))
