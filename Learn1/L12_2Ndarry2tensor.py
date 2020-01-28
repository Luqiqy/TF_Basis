import tensorflow.compat.v1 as tf
import numpy as np
# 创建一维的ndarray
array = np.array([0, 1, 2], np.float32)
# ndarray转换为tensor
t = tf.convert_to_tensor(array, tf.float32, name='t')
print(t)
