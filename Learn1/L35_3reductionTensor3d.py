import tensorflow.compat.v1 as tf
# 构造三维张量
t3d = tf.constant([[[1, 0, 1], [1, 1, 1]],
                   [[2, 0, 2], [2, 2, 0]],
                   [[3, 0, 3], [3, 3, 0]]], tf.float32)
session = tf.Session()
# 三维向量沿行方向‘axis=0’求和
sum0 = tf.reduce_sum(t3d, axis=0)
# 三维张量沿列方向‘axis=1’求和
sum1 = tf.reduce_sum(t3d, axis=1)
# 三维张量沿深度方向‘axis=2’求和
sum2 = tf.reduce_sum(t3d, axis=2)
# 打印结果
print("三维张量：\n", session.run(t3d))
print("沿‘0’轴方向的和sum0：\n", session.run(sum0))
print("沿‘1’轴方向的和sum1：\n", session.run(sum1))
print("沿‘2’轴方向的和sum2：\n", session.run(sum2))
