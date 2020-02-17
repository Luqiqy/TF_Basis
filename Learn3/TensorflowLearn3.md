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

##### 1.2.1 保存模型

Tensorflow保存模型可以理解为用文件保存程序中的**Variable**对象。

* 定义两个**Variable**，并初始化为不同的值，然后声明一个`tf.train.Saver`类的对象，调用该类中的方法***save***，将这两个**Variable**对象保存到model.ckpt的文件中。

```python
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
```

* 使用常用的字典类管理**Variable**对象，保存模型。

```python
import tensorflow.compat.v1 as tf
# 用字典管理变量
weights = {
    'w1': tf.Variable([11, 12, 13], dtype=tf.float32, name='w1'),
    'w2': tf.Variable([21, 22], dtype=tf.float32, name='w2')
    }
bias = {
    'b1': tf.Variable([101, 102], dtype=tf.float32, name='b1'),
    'b2': tf.Variable(2, dtype=tf.float32, name='b2')
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
```

##### 1.2.2 加载模型

* 声明一个`tf.train.Saver`类的对象，调用该类中的方法***restore***，读取model.ckpt文件中变量的值。

  ```python
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
  ```

  > 弊端：使用此种方法要知道要读取的变量的名称及其形状。

* 直接获取文件中的变量的名称及其对应的值。

  ```python
  import tensorflow.compat.v1 as tf
  from tensorflow.python import pywrap_tensorflow
  # 获取在'./L2model/'文件夹下的ckpt文件，并打印获取的文件
  ckpt = tf.train.latest_checkpoint('./L2model/')
  print('获取的ckpt文件：'+ckpt)
  # 创建NewCheckpointReader类，读取ckpt文件中的变量名称及其对应的值
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
      print("tensor_name:", key)
      print(reader.get_tensor(key))
  ```

* 使用常用的字典类管理**Variable**对象，加载模型。

  ```python
  import tensorflow.compat.v1 as tf
  # 用字典类管理变量
  weights = {
      'w1': tf.Variable([1, 13, 22], dtype=tf.float32, name='w1'),
      'w2': tf.Variable([31, 32], dtype=tf.float32, name='w2')
      }
  bias = {
      'b1': tf.Variable([2, 12], dtype=tf.float32, name='b1'),
      'b2': tf.Variable(23, dtype=tf.float32, name='b2')
      }
  # 声明一个tf.train.Saver类
  saver = tf.train.Saver()
  with tf.Session() as sess:
      # 加载modelMul.ckpt文件
      saver.restore(sess, './L2modelMul/modelMul.ckpt')
      # 打印结果
      print(sess.run(weights['w1']))
      print(sess.run(weights['w2']))
      print(sess.run(bias['b1']))
      print(sess.run(bias['b2']))
  sess.close()
  ```

  