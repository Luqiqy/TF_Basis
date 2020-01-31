import tensorflow.compat.v1 as tf
# 构造三维张量
t = tf.constant([[[1, 0, 1], [1, 1, 1]],
                 [[2, 0, 2], [2, 2, 0]],
                 [[3, 0, 3], [3, 3, 0]]], tf.float32)
# 从t的位置"[0, 0, 1]"开始，取范围为2行x2列深度为1的区域的值
y = tf.slice(t, [0, 0, 1], [2, 2, 1])
# 创建会话
session = tf.Session()
print("原张量：\n", session.run(t))
print("访问的张量：\n", session.run(y))
