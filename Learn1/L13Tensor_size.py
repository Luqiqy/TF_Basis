#利用Tensorflow中的函数shape得到张量的尺寸
import tensorflow.compat.v1 as tf
t = tf.constant([[1, 2, 3],
                 [2, 0, 2],
                 [3, 2, 1]], tf.float32)
session = tf.Session()
s = tf.shape(t)
print("Tensor_size:", session.run(s))
