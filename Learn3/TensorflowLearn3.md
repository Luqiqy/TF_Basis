## 三. 回归分析

### 1. 线性回归分析

线性回归分析中只包括一个自变量和一个因变量，且二者的关系可以用一条直线近似表示，这种回归分析称为一元线性回归分析。

回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。

#### 1.1 一元线性回归

假设已知***xoy***二维平面上***N***个点组成的点集**{(x^(i)^, y^(i)^)}** ∈ **R x R**, **i** = 1, 2, 3, … N，求一条直线 ***y = wx+b***，使得这些点沿***y***方向到直线的距离的平方和（即损失函数）最小。

***例：*** 已知***xoy***平面上有6个点(1, 3), (2, 4), (3, 7), ( 4, 8), (5, 11), (6, 14)，寻找一条直线***y = wx+b***, 使得这些点沿y轴方向到该直线的距离的平方和最小。

***解：*** 思路：将要求解的未知量直线的**斜率w**和**截距b**在对应的代码中初始化为两个**Variable**对象，而损失函数可以使用**square**和**reduce_sum**实现，然后针对构造的损失函数利用梯度下降函数**GradientDescentOptimizer**计算直线的**斜率w**和**截距b**。

```python
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# 6个点的横坐标
x = tf.constant([1, 2, 3, 4, 5, 6], tf.float32)
# 6个点的纵坐标
y = tf.constant([3, 4, 7, 8, 11, 14], tf.float32)
# 初始化直线的斜率和截距
w = tf.Variable(1.0, dtype=tf.float32)
b = tf.Variable(1.0, dtype=tf.float32)
# 损失函数，即6个点到直线沿y轴方向距离的平方和
loss = tf.reduce_sum(tf.square(y-(w*x+b)))
# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 对损失函数利用梯度下降法计算w和b
opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
# 记录每次迭代后的平均误差（Mean Squared Error）
MSE = []
# 循环500次
for i in range(500):
    sess.run(opti)
    MSE.append(sess.run(loss))
    # 每隔50词打印直线的斜率和截距
    if i%50==9:
        print((sess.run(w), sess.run(b)))
# 画出损失函数的变化曲线图
plt.figure(1)
plt.plot(MSE)
plt.show()
# 画出6个点和最后计算得到的直线
plt.figure(2)
x_array, y_array = sess.run([x, y])
plt.plot(x_array, y_array, 'o')
xx = np.arange(0, 10, 0.05)
yy = sess.run(w)*xx+sess.run(b)
plt.plot(xx, yy)
plt.show()
```

#### 1.2 保存和加载回归模型

