# ## 数据类型的转换：布尔型转换为数值型
import tensorflow.compat.v1 as tf
# 构造张量
t = tf.constant([[False, True, True],
                 [True, False, True]], tf.bool)
session = tf.Session()
# 数值型转换为布尔型
r = tf.cast(t, tf.float32)
print(session.run(t))
print(session.run(r))