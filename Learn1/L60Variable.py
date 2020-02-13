import tensorflow.compat.v1 as tf
# 创建Variable对象x
v = tf.Variable(tf.constant([2, 3], tf.float32))
session = tf.Session()
# Variable对象初始化
session.run(tf.global_variables_initializer())
print("v初始化的值：\n", session.run(v))
# 使用成员函数assign_add改变本身的值
session.run(v.assign_add([10, 20]))
print("v的当前值：\n", session.run(v))
