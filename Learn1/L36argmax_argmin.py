import tensorflow.compat.v1 as tf
# 构建二维张量
t2d = tf.constant([[0, 2, 5, 6],
                   [7, 4, 9, 1]], tf.float32)
# 沿列方向‘axis=1’，得到最大值的列信息
result_max = tf.argmax(t2d, axis=1)
# 沿列方向‘axis=1’，得到最小值的列信息
result_min = tf.argmin(t2d, axis=1)
session = tf.Session()
# 打印结果
print("最大值位置索引：", session.run(result_max))
print("最小值位置索引：", session.run(result_min))
