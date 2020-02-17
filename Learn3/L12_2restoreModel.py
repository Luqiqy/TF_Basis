import tensorflow.compat.v1 as tf
# 初始化两个变量，变量形状要与model.ckpt中相同
v1 = tf.Variable([11, 12, 13], dtype=tf.float32, name='v1')
v2 = tf.Variable([15, 16], dtype=tf.float32, name='v2')
# 声明一个tf.train.Sever类
saver = tf.train.Saver()
with tf.Session() as sess:
    # 加载./L2/model.ckpt下文件
    saver.restore(sess, './L2model/model.ckpt')
    # 打印两个变量的值
    print(sess.run(v1))
    print(sess.run(v2))
sess.close()