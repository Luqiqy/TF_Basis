import tensorflow.compat.v1 as tf
# 创建二维张量，充当被加张量、被减张量、被乘张量、被除张量
t1 = tf.constant([[0, 1, 2],
                  [3, 4, 5]], tf.float32)
# 创建与t1同类型的张量
# t2 = tf.constant([[5, 3, 1]], tf.float32)    # 与t1列数相同则按行计算
t2 = tf.constant([[1],
                  [2]], tf.float32)            # 与t1行数相同则按列计算
session = tf.Session()
# 计算两个二维张量相加
result_add = tf.add(t1, t2)                    # 等价于result_add = t1+t2
# 计算两个二维向量相减
result_subtract = tf.subtract(t1, t2)          # 等价于result_subtract = t1-t2
# 计算两个二维向量相乘
result_multiply = tf.multiply(t1, t2)          # 等价于result_multiply = t1*t2
# 计算一个标量与一个张量相乘
result_scalar_mul = tf.scalar_mul(2, t1)       # 等价于result_scalar_mul = 2*t1
# 计算两个二维张量相除
result_div = tf.div(t1, t2)                    # 等价于result_div = t1/t2
# 打印结果
print("二维张量t1：\n", session.run(t1))
print("二维张量t2：\n", session.run(t2))
print("相加结果result_add：\n", session.run(result_add))
print("相减结果result_subtract：\n", session.run(result_subtract))
print("相乘结果result_multiply：\n", session.run(result_multiply))
print("标量2与张量t1相乘结果result_scalar_mul：\n", session.run(result_scalar_mul))
print("相除结果result_div：\n", session.run(result_div))