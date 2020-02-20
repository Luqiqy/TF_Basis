import tensorflow.compat.v1 as tf
# 变量x
x = tf.Variable(tf.constant([[2, 1, 3]], tf.float32))
w = tf.constant([[2], [3], [4]], tf.float32)
# 函数g
g = tf.matmul(x, w)
# 函数f=leaky_relu(g)
f = tf.nn.leaky_relu(g, alpha=0.2)
# 牛顿梯度下降法
opti = tf.train.GradientDescentOptimizer(0.5).minimize(f)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 打印结果
for i in range(5):
    sess.run(opti)
    print('第%d次迭代的值'%(i+1))
    print(sess.run(x))
