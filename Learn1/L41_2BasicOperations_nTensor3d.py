import tensorflow.compat.v1 as tf
# 构建2行2列2深度的三维张量
t3d = tf.constant([[[2, 5], [4, 3]],
                   [[6, 1], [1, 2]]], tf.float32)
# 构建2行2列2深度的三维张量
t222 = tf.constant([[[1, 2], [2, 3]],
                    [[3, 2], [5, 3]]], tf.float32)
# 构建1行2列2深度的三维张量
t122 = tf.constant([[[1, 4], [3, 2]]], tf.float32)
# 构建2行1列2深度的三维张量
t212 = tf.constant([[[1, 2]],
                    [[3, 4]]], tf.float32)
# 构建1行1列2深度的三维张量
t112 = tf.constant([[[1, 4]]], tf.float32)
session = tf.Session()
# 打印结果
print("三维张量t3d：\n", session.run(t3d))
print("t3d+t222: \n", session.run(t3d+t222))
print("t3d+t122: \n", session.run(t3d+t122))
print("t3d+t212: \n", session.run(t3d+t212))
print("t3d+t112: \n", session.run(t3d+t112))
