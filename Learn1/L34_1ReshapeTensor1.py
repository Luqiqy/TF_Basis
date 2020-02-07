import tensorflow.compat.v1 as tf
# 创建张量
t = tf.constant([[[0, 1], [2, 3], [4, 5]],
                 [[6, 7], [8, 9], [10, 11]]], tf.float32)
session = tf.Session()
# 使用reshape函数改变张量形状为4行1列3深度
t1 = tf.reshape(t, [4, 1, -1])  # 参数-1表示此维度默认由计算得出
# 打印输出原张量及其形状
print("张量原形状：\n", session.run(t))
print("张量的形状：", t.get_shape())
# 打印改变形状后的张量及其形状
print("张量形状改变后：\n", session.run(t1))
print("张量的形状：\n", t1.get_shape())