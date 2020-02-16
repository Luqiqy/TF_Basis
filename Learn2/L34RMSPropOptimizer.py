import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# RMSProp梯度下降
opti = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, epsilon=1e-10).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))