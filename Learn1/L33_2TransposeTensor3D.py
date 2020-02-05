import tensorflow.compat.v1 as tf
# 创建三维张量
x = tf.constant([[[1, 2], [3, 4], [5, 6]],
                 [[7, 8], [9, 0], [1, 0]]])
session = tf.Session()
# 使用函数transpose对张量进行转置，
# perm中'0'是沿行方向，'1'是沿列方向，‘2’是沿深度方向
# #对每一行即每一个"(1, 2)平面"进行转置
r0 = tf.transpose(x, perm=[0, 2, 1])
# #对每一列即每一个"(0, 2)平面"进行转置
r1 = tf.transpose(x, perm=[2, 1, 0])
# #对每一深度即每一个"(0, 1)平面"进行转置
r2 = tf.transpose(x, perm=[1, 0, 2])
print("未转置的三维张量：\n", session.run(x))
print("转置后的三维张量：\n", session.run(r0))
print("转置后的三维张量：\n", session.run(r1))
print("转置后的三维张量：\n", session.run(r2))