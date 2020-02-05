import tensorflow.compat.v1 as tf
# 创建二维张量
x = tf.constant([[1, 2, 3],
                 [4, 5, 6]], tf.float32)
session = tf.Session()
# 使用函数transpose进行张量转置
r = tf.transpose(x, perm=[1, 0])

print("转置前的二维张量：\n", session.run(x))
print("转置后的二维张量：\n", session.run(r))
