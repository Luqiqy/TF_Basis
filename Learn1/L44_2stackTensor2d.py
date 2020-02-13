import tensorflow.compat.v1 as tf
# 构建二维张量t1
t1 = tf.constant([[11, 12, 13],
                  [14, 15, 16],
                  [17, 18, 19]], tf.float32)
# 构建二维张量t2
t2 = tf.constant([[4, 5, 6],
                  [7, 8, 9],
                  [1, 3, 2]], tf.float32)
# 沿‘0’方向进行堆叠
tt0 = tf.stack([t1, t2], axis=0)
# 沿‘1’方向进行堆叠
tt1 = tf.stack([t1, t2], axis=1)
# 沿‘2’方向进行堆叠
tt2 = tf.stack([t1, t2], axis=2)
session = tf.Session()
# 打印结果
print("沿‘0’方向进行堆叠:\n", session.run(tt0))
print("tt0的shape：", tt0.get_shape())
print("沿‘1’方向进行堆叠:\n", session.run(tt1))
print("tt1的shape：", tt0.get_shape())
print("沿‘2’方向进行堆叠:\n", session.run(tt2))
print("tt2的shape：", tt0.get_shape())