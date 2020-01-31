import tensorflow as tf
t0 = tf.constant(3, tf.float32)
print(t0)
t1 = tf.constant([1, 2, 3], tf.float32)
print(t1)
t2 = tf.constant([[1, 2, 3],
                  [2, 0, 2],
                  [3, 2, 1]], tf.float32)
print(t2)
t30 = tf.constant([[[1, 0, 1],
                    [0, 1, 0]],
                   [[2, 0, 2],
                    [0, 2, 3]]], tf.float32)
print(t30)
t31 = tf.constant([[[1, 2, 1], [0, 0, 0]],
                   [[0, 0, 0], [1, 2, 1]]], tf.float32)
print(t31)
t4 = tf.constant([
                  [[[1, 2, 3], [3, 2, 1]],
                   [[3, 2, 1], [1, 2, 3]]],
                  [[[1, 0, 1], [2, 0, 1]],
                   [[1, 0, 2], [2, 0, 2]]]
                 ], tf.float32)
print(t4)