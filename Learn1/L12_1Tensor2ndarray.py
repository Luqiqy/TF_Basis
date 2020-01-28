#方法一
'''
import tensorflow.compat.v1 as tf
#import tensorflow as tf
#创建一维张量
t = tf.constant([0, 1, 2], tf.float32)
#创建会话
session = tf.Session()
#张量转换为ndrarray
array = session.run(t)
print(array)
'''
#方法二：先创建会话，再利用Tensor的成员函数eval将Tensor转换为ndarray
import tensorflow.compat.v1 as tf
#import tensorflow as tf
#创建一维张量
t = tf.constant([0, 1, 2], tf.float32)
#创建会话
session = tf.Session()
#张量转换为ndrarray
array = t.eval(session = session)
print(array)
