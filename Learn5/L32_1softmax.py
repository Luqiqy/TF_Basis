import tensorflow.compat.v1 as tf
# 输入张量
x = tf.constant([[2, 1, 2],
                 [2, 2, 2]], tf.float32)
# 分别对每一行（axis=1）进行softmax处理
s = tf.nn.softmax(x, axis=1)
# 打印结果
session = tf.Session()
print(session.run(s))
