import tensorflow.compat.v1 as tf
# 构建三维张量t1
t1 = tf.constant([[[9, 2], [1, 9]],
                  [[0, 9], [9, 9]]], tf.float32)
# 构建三维张量t2
t2 = tf.constant([[[5, 5], [5, 5]],
                  [[5, 5], [5, 5]]], tf.float32)
# 沿行方向“axis=0”进行连接
tt0 = tf.concat([t1, t2], axis=0)
# 沿列方向“axis=1”进行连接
tt1 = tf.concat([t1, t2], axis=1)
# 延深度方向“axis=2”进行连接
tt2 = tf.concat([t1, t2], axis=2)
session = tf.Session()
# 打印结果
print("三维张量t1:\n", session.run(t1))
print("三维张量t2:\n", session.run(t2))
print("沿‘0’方向连接：\n", session.run(tt0))
print("沿‘1’方向连接：\n", session.run(tt1))
print("沿‘2’方向连接：\n", session.run(tt2))