import tensorflow.compat.v1 as tf
# 构造一维张量t1
t1 = tf.constant([1, 2, 3], tf.float32)
# 构造一维张量t2
t2 = tf.constant([4, 5, 6], tf.float32)
# 沿行方向‘axis=0’进行堆叠
tt0 = tf.stack([t1, t2], axis=0)
# 沿列方向‘axis=1’进行堆叠
tt1 = tf.stack([t1, t2], axis=1)
session = tf.Session()
# 打印结果
print("沿‘0’方向堆叠：\n", session.run(tt0))
print("沿‘1’方向堆叠：\n", session.run(tt1))

