import tensorflow.compat.v1 as tf
# 创建1维张量
t1d = tf.constant([0, 1, 2, 3, 4], tf.float32)
# 使用reduce_sum计算张量的和
sum0 = tf.reduce_sum(t1d)
# 使用reduce_mean计算张量的平均值
mean0 = tf.reduce_mean(t1d)
# 使用reduce_max计算张量的最大值
max0 = tf.reduce_max(t1d)
# 使用reduce_min计算张量的最小值
min0= tf.reduce_min(t1d)
session = tf.Session()
# 打印结果
print("张量：", session.run(t1d))
print("张量的和：", session.run(sum0))
print("张量的平均值：", session.run(mean0))
print("张量的最大值：", session.run(max0))
print("张量的最小值：", session.run(min0))