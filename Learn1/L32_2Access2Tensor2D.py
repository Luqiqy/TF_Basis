import tensorflow.compat.v1 as tf
# 构造二维张量
t = tf.constant([[0, 1, 2],
                 [3, 4, 5]], tf.float32)
# 从t的位置"[0, 1]"开始，取范围为2行x2列的区域的值
y = tf.slice(t, [0, 1], [2, 2])
# 创建会话
session = tf.Session()
print("原张量：\n", session.run(t))
print("访问的张量：\n", session.run(y))
