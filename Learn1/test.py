import tensorflow.compat.v1 as tf
import numpy as np
a = tf.constant([[[1],[2]],
                 [[2],[1]]])
print(a.get_shape())
session = tf.Session()
array = session.run(a)
print(array)
print(np.shape(array))