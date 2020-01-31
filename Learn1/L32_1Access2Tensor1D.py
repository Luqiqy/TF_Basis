import tensorflow.compat.v1 as tf
# 构造一维张量
t = tf.constant([0, 1, 2, 3, 4], tf.float32)
# 从t的第一个位置"[1]"开始，取长度为3的区域的值
y = tf.slice(t, [1], [3])
# 创建会话
session = tf.Session()
print("原张量：", session.run(t))
print("访问的张量：", session.run(y))