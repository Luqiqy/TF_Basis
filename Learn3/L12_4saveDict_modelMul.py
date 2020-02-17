import tensorflow.compat.v1 as tf
# 用字典管理变量
weights = {
    'w1':tf.Variable([11, 12, 13], dtype=tf.float32, name='w1'),
    'w2':tf.Variable([21, 22], dtype=tf.float32, name='w2')
    }
bias = {
    'b1':tf.Variable([101, 102], dtype=tf.float32, name='b1'),
    'b2':tf.Variable(2, dtype=tf.float32, name='b2')
    }

# 创建会话
sess = tf.Session()
# 声明一个tf.train.Saver类
saver = tf.train.Saver()
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 将变量存在./L2modelMul/modelMul.ckpt
    saver.save(sess, './L2modelMul/modelMul.ckpt')