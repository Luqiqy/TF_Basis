import tensorflow.compat.v1 as tf
# 创建二维张量，‘0’代表行方向，‘1’代表列方向
t2d = tf.constant([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 0]], tf.float32)
session = tf.Session()
# 二维向量沿行方向‘axis=0’求和
sum0 = tf.reduce_sum(t2d, axis=0)
# 二维张量沿列方向‘axis=1’求和
sum1 = tf.reduce_sum(t2d, axis=1)
# 打印结果
print("二维张量：\n", session.run(t2d))
print("沿‘0’轴方向的和sum0：", session.run(sum0))
print("沿‘1’轴方向的和sum1：", session.run(sum1))
