'''
'# 利用Tensorflow中的求和函数reduce_sum和函数tf.nn.softmax定义softmax损失函数为tf.reduce_sum(-y*tf.nn.softmax(_y,1))
import tensorflow.compat.v1 as tf
# 假设_y为全连接神经网络的输出（输出层有3个神经元）
_y = tf.constant([[0, 2, -3], [4, -5, 6]], tf.float32)
# 人工分类结果
y = tf.constant([[1, 0, 0], [0, 0, 1]], tf.float32)
# 计算softmax熵
_y_softmax = tf.nn.softmax(_y)
entropy = tf.reduce_sum(-y*tf.log(_y_softmax), 1)
# loss损失函数
loss = tf.reduce_sum(entropy)
# 打印结果
session = tf.Session()
print(session.run(loss))
'''
# Tensorflow通过softmax_cross_entropy_with_logits_v2可以直接得到softmax熵
import tensorflow.compat.v1 as tf
# 假设_y为全连接神经网络的输出（输出层有3个神经元）
_y = tf.constant([[0, 2, -3], [4, -5, 6]], tf.float32)
# 人工分类结果
y = tf.constant([[1, 0, 0], [0, 0, 1]], tf.float32)
# 计算softmax熵
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_y, labels=y)
# loss损失函数
loss = tf.reduce_sum(entropy)
# 打印结果
session = tf.Session()
print(session.run(loss))
