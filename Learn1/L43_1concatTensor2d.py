import tensorflow.compat.v1 as tf
# 构建待拼接的二维张量t1
t1 = tf.constant([[1, 2, 3],
                  [4, 5, 6]], tf.float32)
# 构建待拼接的二维张量t2
t2 = tf.constant([[4, 5],
                  [7, 8]], tf.float32)
# 利用函数concat来实现对t1和t2的拼接
t = tf.concat([t1, t2], axis=1)
session = tf.Session()
# 打印结果
print("拼接后的张量：\n", session.run(t))