import tensorflow.compat.v1 as tf
import numpy as np
# 自变量(x, y)的值
xy = tf.placeholder(tf.float32, [None, 2])
# 初始化z=w1*x+w2*y+b中的w1,w2,b
w = tf.Variable(tf.constant([[1], [1]], tf.float32), dtype=tf.float32, name='w')
b = tf.Variable(1.0, dtype=tf.float32, name='b')
# 超平面
z = tf.matmul(xy, w)+b
# 声明一个tf.train.Saver类
saver = tf.train.Saver()
# 从文件夹中获取最近的ckpt文件，并打印名称
ckpt = tf.train.latest_checkpoint('./L3model/')
print("获得的ckpt文件："+ckpt)
# 创建会话
sess = tf.Session()
# 加载ckpt文件中的变量w, b的值
saver.restore(sess, ckpt)
# 计算在坐标(6, 7)和(8, 10)处的值
pred = sess.run(z, feed_dict={xy: np.array([[6, 7], [8, 10]], np.float32)})
print("在坐标(6, 7)和(8, 10)处的值：\n", pred)
