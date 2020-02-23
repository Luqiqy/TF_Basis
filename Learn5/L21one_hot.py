import tensorflow.compat.v1 as tf
# depth=10代表有10个类别，因此返回张量有10列，axis=1代表按照每一行存储类别的数字化
v = tf.one_hot([0, 1, 9, 2, 8, 3, 4, 7, 6, 5], depth=10, axis=1, dtype=tf.float32)
session = tf.Session()
print(session.run(v))