import tensorflow.compat.v1 as tf
# 第一个variable，初始化为一个长度为3的一维张量
v1 = tf.Variable(tf.constant([1, 2, 3], tf.float32), dtype=tf.float32, name='v1')
# 第一个variable，初始化为一个长度为2的一维张量
v2 = tf.Variable(tf.constant([4, 5], tf.float32), dtype=tf.float32, name='v2')
# 声明一个tf.train.Saver对象
saver = tf.train.Saver()
# 创建会话，初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 将变量v1和v2保存到./L2/model.ckpt路径下
save_path = saver.save(sess, './L2model/model.ckpt')
sess.close()