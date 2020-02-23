import tensorflow.compat.v1 as tf
# 输出层的值
logits = tf.constant([[-8.29, 0.64, 9.22, -0.08, 5.6, 6.15, -10.21, 7.51, 7.73, 9.43]],
                     tf.float32)
# 人工分类的标签
labels = tf.constant([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], tf.float32)
# sigmoid交叉熵
entroy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
# 损失值
loss = tf.reduce_sum(entroy)
# 打印损失值
session = tf.Session()
print(session.run(loss))