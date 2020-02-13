import tensorflow.compat.v1 as tf
# 构建二维张量x
x = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], tf.float32)
# 构建二维张量y
y = tf.constant([[4, 5, 3],
                 [7, 5, 9],
                 [1, 3, 2]], tf.float32)
# 逐个元素进行对比
r = tf.equal(x, y)
session = tf.Session()
# 打印结果
print("张量x：\n", session.run(x))
print("张量y：\n", session.run(y))
print("对比结果：\n", session.run(r))




