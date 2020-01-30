# ##数据类型的转换：数值型转换为布尔型
import tensorflow.compat.v1 as tf
# 构造张量
t = tf.constant([[1, 0, 3],
                 [0, 2, 0]], tf.float32)
session = tf.Session()
# 数值型转换为布尔型
r = tf.cast(t, tf.bool)
print(t.eval(session = session))
print(session.run(r))