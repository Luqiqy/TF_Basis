# Tensorflow基础

## 基本的数据结构及运算

### 1. 张量Tensor

#### 1.1 张量Tensor的定义

构造张量提供的函数tensorflow.constant：

```python
tensorflow.constant(
    value,				# value是必须的，可以是一个数值，也可以是一个列表
    dtype=None,			# 数据类型
    shape=Nope,			# 张量的尺寸
    name=“Const”,		# name可以是任何内容
    verufy_shape=False	# 默认为False，如果修改为True的话表示检查value的形状与shape是否相符
	)
```

* 零维张量

  ```python
  t = tf.constant(0, tf.float32)
  #output:Tensor("Const:0", shape=(), dtype=float32) 并未打印出张量的值，而是打印出了张量的信息
  ```

* 一维张量

  ```python
  t = tf.constant([0,1,2], tf.float32)
  #output:Tensor("Const_1:0", shape=(3,), dtype=float32)
  ```

* 二维张量

  ```python
  t = tf.constant([[1, 2, 3],
                   [2, 0, 2], 
                   [3, 2, 1]], tf.float32)
  #output:Tensor("Const_2:0", shape=(3, 3), dtype=float32)
  ```

* 三维张量

  ```python
  t0 = tf.constant([[[1, 0],
                     [0, 1]],
                    [[2, 0],
                     [0, 2]]], tf.float32)
  print(t0)
  #output:Tensor("Const_4:0", shape=(2, 2, 2), dtype=float32)
  t1 = tf.constant([[[1, 2], [0, 0]],
                    [[0, 0], [1, 2]]], tf.float32)
  print(t1)
  #output:Tensor("Const_4:0", shape=(2, 2, 2), dtype=float32)
  ```

* 四维张量

  ```python
  t = tf.constant([
                   [[[1, 2, 3], [3, 2, 1]],
                    [[3, 2, 1], [1, 2, 3]]],
                   [[[1, 0, 1], [2, 0, 1]],
                    [[1, 0, 2], [2, 0, 2]]]
                  ], tf.float32)
  #output:Tensor("Const_5:0", shape=(2, 2, 2, 3), dtype=float32)
  ```

#### 1.2 Tensor与ndarray转换

##### 1.2.1 Tensor转换为ndarray

* ```python
  import tensorflow.compat.v1 as tf
  #import tensorflow as tf
  #创建一维张量
  t = tf.constant([0, 1, 2], tf.float32)
  #创建会话
  session = tf.Session()
  #张量转换为ndrarray
  array = session.run(t)
  print(array)
  ```

* ```python
  #先创建会话，再利用Tensor的成员函数eval将Tensor转换为ndarray
  import tensorflow.compat.v1 as tf
  #import tensorflow as tf
  #创建一维张量
  t = tf.constant([0, 1, 2], tf.float32)
  #创建会话
  session = tf.Session()
  #张量转换为ndrarray
  array = t.eval(session = session)
  print(array)
  ```

##### 1.2.2 ndarray转换为Tensor

使用函数tensorflow.convert_to_tensor()

```python
tf.convert_to_tensor(
    value,					# 能够转换为tensor的类型值
    dtype=None,
    name=None,
    preferred_dtype=None,
    dtype_hint=None
)
```

```python
import tensorflow.compat.v1 as tf
import numpy as np
#创建一维的ndarray
array = np.array([0, 1, 2], np.float32)
#ndarry转换为tensor
t = tf.convert_to_tensor(array, tf.float32, name = 't')
print(t)
```

#### 1.3 张量的尺寸

* 利用Tensorflow中的函数shape得到张量的尺寸

  ```python
  tensorflow.shape(				# 返回out_type类型的张量
      		input,				# 张量或稀疏张量
      		name=None,
      		out_type=tf.int32	# 默认为tf.int32
  		)
  ```

  ```python
  #利用Tensorflow中的函数shape得到张量的尺寸
  import tensorflow.compat.v1 as tf
  t = tf.constant([[1, 2, 3],
                   [2, 0, 2],
                   [3, 2, 1]], tf.float32)
  session = tf.Session()
  s = tf.shape(t)
  print("Tensor_size:", session.run(s))
  ```

* 利用成员函数get_shape()或成员变量shape得到张量的尺寸

  ```python
  # 利用成员函数get_shape()或成员变量shape得到张量的尺寸
  import tensorflow.compat.v1 as tf
  t = tf.constant([[1, 2, 3],
                   [2, 0, 2],
                   [3, 2, 1]], tf.float32)
  #s = t.shape
  #print(s, type(s))
  
  a = t.get_shape()
  print(a, type(a))
  ```

#### 1.4 图像转换为张量

```python
import tensorflow.compat.v1 as tf
# 读取图像文件
image = tf.read_file("lena512.bmp", 'r')
# 将图像文件解码成Tensor
image_tensor = tf.image.decode_bmp(image)
# 图像张量的尺寸
shape = tf.shape(image_tensor)
session = tf.Session()
print("图像的形状：")
print(session.run(shape))
# Tensor转换为ndarray
image_ndarray = image_tensor.eval(session=session)
print(image_ndarray)
```

### 2. 随机数

#### 2.1 均匀分布随机数

均匀分布的概率密度函数：
$$
p(X=x)=\left\{\begin{matrix}
\frac{1}{b-a},\:\:a\leqslant x< b
\\ 
0,\:\:\:\:\:\:\:\:\:其他
\end{matrix}\right.
$$

```python
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
# 构造四维张量，即10个深度为3的4行5列的三维张量，每个值在区间[0,10]满足均匀分布
x = tf.random_uniform([10, 3, 4, 5], minval=0, maxval=10, dtype=tf.float32)
# 查看该四维张量的尺寸
print(x.get_shape())
# 将该Tensor转换为nddarry
session = tf.Session()
array = x.eval(session=session)
# 查看array的shape
print(np.shape(array))
print(type(array), tf.shape(array))
# 将array转化为1个一维的ndarray来画图
array1d=array.reshape([-1])
plt.hist(array1d)
plt.show()
```

#### 2.2 正态分布随机数

一维正态分布的概率密度函数：
$$
P(X=x)=\frac{1}{\sqrt{2\pi }\sigma }e\tfrac{-(x-\mu )^{2}}{2\sigma ^{2}}
$$

```python
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import math
# 正态分布标准差sigma=1，均值mu=10
sigma = 1
mu = 10
# 构造四位张量，即10个深度为3行为4列为5的三维向量，张量中的值满足均值为10，标准差为1的正态分布
result = tf.random_normal([10, 3, 4, 5], mu, sigma, tf.float32)
# 将Tensor转化为ndarray
session = tf.Session()
array = session.run(result)
# 将array转换为1个一维的ndarray
array1d = array.reshape([-1])
# 计算并显示直方图
histogram, bins, patch = plt.hist(array1d, 25, facecolor='gray', alpha=0.5, normed=True)
x = np.arange(5, 15, 0.01)
y = 1.0/(math.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x-mu, 2.0)/(2*math.pow(sigma,2)))
plt.plot(x, y)
plt.show()
```

### 3. 单个张量的运算

#### 3.1 改变张量的数据类型

Tensorflow通过函数cast实现数据类型的转换。

```python
tensorflow.cast(x, dtype, name=None)
```

##### 3.1.1 数值型转换为布尔型

```python
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
```

##### 3.1.2 布尔型转换为数值型

```python
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
```

#### 3.2 访问张量

Tensorflow通过函数slice访问张量中任意一个区域的值。

```python
tensorflow.slice(input_, begin, size, name=None)
```

##### 3.2.1 访问一维张量

```python
import tensorflow.compat.v1 as tf
# 构造一维张量
t = tf.constant([0, 1, 2, 3, 4], tf.float32)
# 从t的第一个位置"[1]"开始，取长度为3的区域的值
y = tf.slice(t, [1], [3])
# 创建会话
session = tf.Session()
print("原张量：", session.run(t))
print("访问的张量：", session.run(y))
```

##### 3.2.2 访问二维张量

```python
import tensorflow.compat.v1 as tf
# 构造二维张量
t = tf.constant([[0, 1, 2],
                 [3, 4, 5]], tf.float32)
# 从t的位置"[0, 1]"开始，取范围为2行x2列的区域的值
y = tf.slice(t, [0, 1], [2, 2])
# 创建会话
session = tf.Session()
print("原张量：\n", session.run(t))
print("访问的张量：\n", session.run(y))
```

##### 3.2.3 访问三维张量

```python
import tensorflow.compat.v1 as tf
# 构造三维张量
t = tf.constant([[[1, 0, 1], [1, 1, 1]],
                 [[2, 0, 2], [2, 2, 0]],
                 [[3, 0, 3], [3, 3, 0]]], tf.float32)
# 从t的位置"[0, 0, 1]"开始，取范围为2行x2列深度为1的区域的值
y = tf.slice(t, [0, 0, 1], [2, 2, 1])
# 创建会话
session = tf.Session()
print("原张量：\n", session.run(t))
print("访问的张量：\n", session.run(y))
```

#### 3.3 转置

Tensorflow通过函数transpose实现张量的转置。

```python
tensorflow.transpose(a, perm=None, name='transpose')	# 将a按照perm转换
```

##### 3.3.1 二维张量转置

```python
import tensorflow.compat.v1 as tf
# 创建二维张量
x = tf.constant([[1, 2, 3],
                 [4, 5, 6]], tf.float32)
session = tf.Session()
# 使用函数transpose进行张量转置
r = tf.transpose(x, perm=[1, 0])
print("转置前的二维张量：\n", session.run(x))
print("转置后的二维张量：\n", session.run(r))
```

##### 3.3.2 三维张量转置 

```python
import tensorflow.compat.v1 as tf
# 创建三维张量
x = tf.constant([[[1, 2], [3, 4], [5, 6]],
                 [[7, 8], [9, 0], [1, 0]]])
session = tf.Session()
# 使用函数transpose对张量进行转置，
# perm中'0'是沿行方向，'1'是沿列方向，‘2’是沿深度方向
# #对每一行即每一个"(1, 2)平面"进行转置
r0 = tf.transpose(x, perm=[0, 2, 1])
# #对每一列即每一个"(0, 2)平面"进行转置
r1 = tf.transpose(x, perm=[2, 1, 0])
# #对每一深度即每一个"(0, 1)平面"进行转置
r2 = tf.transpose(x, perm=[1, 0, 2])
print("未转置的三维张量：\n", session.run(x))
print("转置后的三维张量：\n", session.run(r0))
print("转置后的三维张量：\n", session.run(r1))
print("转置后的三维张量：\n", session.run(r2))
```

#### 3.4 改变形状

Tensorflow通过函数reshape改变张量的形状。

##### 3.4.1 改变张量不改变维度

```python
import tensorflow.compat.v1 as tf
# 创建张量
t = tf.constant([[[0, 1], [2, 3], [4, 5]],
                 [[6, 7], [8, 9], [10, 11]]], tf.float32)
session = tf.Session()
# 使用reshape函数改变张量形状
t1 = tf.reshape(t, [4, 1, -1])	# 参数-1表示此维度默认由计算得出
# 打印输出原张量及其形状
print("张量原形状：\n", session.run(t))
print("张量的形状：", t.get_shape())
# 打印改变形状后的张量及其形状
print("张量形状改变后：\n", session.run(t1))
print("张量的形状：\n", t1.get_shape())
```

##### 3.4.2 改变张量改变维度

```python
import tensorflow.compat.v1 as tf
# 创建张量
t = tf.constant([[[0, 1], [2, 3], [4, 5]],
                 [[6, 7], [8, 9], [10, 11]]], tf.float32)
session = tf.Session()
# 使用reshape改变张量形状为2行6列
t2 = tf.reshape(t, [2, -1])
# 打印输出原张量及其形状
print("张量原形状：\n", session.run(t))
print("张量的形状：", t.get_shape())
# 打印改变形状后的张量及其形状
print("张量形状改变后：\n", session.run(t2))
print("张量的形状：\n", t2.get_shape())
```

#### 3.5 归约运算

归约运算（reduction）经常被用来表示在大规模数据集中查寻数据集的总和、平均值、最大值、最小值等问题。Tensorflow中以reduce开头的函数接口代表归约运算，如reduce_sum用于计算张量的和、reduce_mean用于计算张量的平均值、reduce_max用于计算张量的最大值，reduce_mix用于计算张量的最小值，其他函数类似。使用方法以求和运算reduce_sum为例。

```python
tensorflow.reduce_sum(
    				input_tensor, 			# input_tensor：待求和的tensor;
    				axis=None, 				# axis：指定的维，如果不指定，则计算所有元素的总和;
    				keepdims=None,			
    				name=None,				# name：操作的名称;
    				reduction_indices=None, # 已弃用；
    				keep_dims=None)			# 已弃用；
	# keepdims：是否保持原有张量的维度，设置为True，结果保持输入tensor的形状，设置为False，结果会降低维度，如果不传入这个参数，则系统默认为False;	
```

##### 3.5.1 一维张量的归约运算

```python
import tensorflow.compat.v1 as tf
# 创建1维张量
t1d = tf.constant([0, 1, 2, 3, 4], tf.float32)
# 使用reduce_sum计算张量的和
sum0 = tf.reduce_sum(t1d)
# 使用reduce_mean计算张量的平均值
mean0 = tf.reduce_mean(t1d)
# 使用reduce_max计算张量的最大值
max0 = tf.reduce_max(t1d)
# 使用reduce_min计算张量的最小值
min0= tf.reduce_min(t1d)
session = tf.Session()
# 打印结果
print("张量：", session.run(t1d))
print("张量的和：", session.run(sum0))
print("张量的平均值：", session.run(mean0))
print("张量的最大值：", session.run(max0))
print("张量的最小值：", session.run(min0))
```

##### 3.5.2 二维张量的归约运算

```python
import tensorflow.compat.v1 as tf
# 创建二维张量，‘0’代表行方向，‘1’代表列方向
t2d = tf.constant([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 0]], tf.float32)
session = tf.Session()
# 以求和为例
# 二维向量沿行方向‘axis=0’求和
sum0 = tf.reduce_sum(t2d, axis=0)
# 二维张量沿列方向‘axis=1’求和
sum1 = tf.reduce_sum(t2d, axis=1)
# 打印结果
print("二维张量：\n", session.run(t2d))
print("沿‘0’轴方向的和sum0：", session.run(sum0))
print("沿‘1’轴方向的和sum1：", session.run(sum1))
```

##### 3.5.3 三维张量的规约运算

```python
import tensorflow.compat.v1 as tf
# 构造三维张量
t3d = tf.constant([[[1, 0, 1], [1, 1, 1]],
                   [[2, 0, 2], [2, 2, 0]],
                   [[3, 0, 3], [3, 3, 0]]], tf.float32)
session = tf.Session()
# 三维向量沿行方向‘axis=0’求和
sum0 = tf.reduce_sum(t3d, axis=0)
# 三维张量沿列方向‘axis=1’求和
sum1 = tf.reduce_sum(t3d, axis=1)
# 三维张量沿深度方向‘axis=2’求和
sum2 = tf.reduce_sum(t3d, axis=2)
# 打印结果
print("三维张量：\n", session.run(t3d))
print("沿‘0’轴方向的和sum0：\n", session.run(sum0))
print("沿‘1’轴方向的和sum1：\n", session.run(sum1))
print("沿‘2’轴方向的和sum2：\n", session.run(sum2))
```

#### 3.6 最值的位置索引

Tensorflow通过函数argmax和argmin实现计算张量中最大和最小值的位置索引。

```python
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
```

### 4. 多个张量之间运算

多个张量之间的运算有加、减、乘、除四种基本运算，分别可用函数tensorflow.add()，tensorflow.subtract()，tensorflow.multiply()，tensorflow.div()来实现，也可直接用运算符号+、-、*、/。

以函数tensorflow.add()来展示函数参数

```python
tensorflow.add(				# 返回值为一个张量x+y，类型同x
	   			x,			# 一个张量
       			y,			# 一个张量
       			name=None)	# 操作的名字
```

#### 4.1 加减乘除基本运算

##### 4.1.1 二维张量基本运算

```python
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
result_add = tf.add(t1, t2)
# 计算两个二维向量相减
result_subtract = tf.subtract(t1, t2)
# 计算两个二维向量相乘
result_multiply = tf.multiply(t1, t2)
# 计算一个标量与一个张量相乘
result_scalar_mul = tf.scalar_mul(2, t1)
# 计算两个二维张量相除
result_div = tf.div(t1, t2)
# 打印结果
print("二维张量t1：\n", session.run(t1))
print("二维张量t2：\n", session.run(t2))
print("相加结果result_add：\n", session.run(result_add))
print("相减结果result_subtract：\n", session.run(result_subtract))
print("相乘结果result_multiply：\n", session.run(result_multiply))
print("标量2与张量t1相乘结果result_scalar_mul：\n", session.run(result_scalar_mul))
print("相除结果result_div：\n", session.run(result_div))
```

针对二维张量进行基本的加减乘除运算有以下总结：

* **M行N列**的二维张量可以与另一个**M行N列**的二维张量相运算
* **M行N列**的二维张量可以与另一个**M行1列**的二维张量相运算
* **M行N列**的二维张量可以与另一个**1行N列**的二维张量相运算，这种情况下，**1行N列**的二维张量可以简化为一个长度为**N**的**一维向量

##### 4.1.2 三维张量基本运算

```python
import tensorflow.compat.v1 as tf
# 构建2行2列2深度的三维张量
t3d = tf.constant([[[2, 5], [4, 3]],
                   [[6, 1], [1, 2]]], tf.float32)
# 构建2行2列2深度的三维张量
t222 = tf.constant([[[1, 2], [2, 3]],
                    [[3, 2], [5, 3]]], tf.float32)
# 构建1行2列2深度的三维张量
t122 = tf.constant([[[1, 4], [3, 2]]], tf.float32)
# 构建2行1列2深度的三维张量
t212 = tf.constant([[[1, 2]],
                    [[3, 4]]], tf.float32)
# 构建1行1列2深度的三维张量
t112 = tf.constant([[[1, 4]]], tf.float32)
session = tf.Session()
# 打印结果
print("三维张量t3d：\n", session.run(t3d))
print("t3d+t222: \n", session.run(t3d+t222))
print("t3d+t122: \n", session.run(t3d+t122))
print("t3d+t212: \n", session.run(t3d+t212))
print("t3d+t112: \n", session.run(t3d+t112))
```

总结：假如有**H行W列D深度**的三维向量，我们分别为每一深度加一个不同的值，就需要**D**个值。最简单的实现方法就是把这**D**个值存储在一个长度为**D**的二维张量中。

#### 4.2 矩阵乘法

Tensorflow提供了关于矩阵（二维张量）乘法的函数matmul。

```python
tensorflow.matmul(
    				a, 						# 一个秩 > 1 的张量
    				b,						# 一个类型跟张量a相同的张量
    				transpose_a=False,		# 如果为真, a则在进行乘法计算前进行转置
    				transpose_b=False,		# 如果为真, b则在进行乘法计算前进行转置
    				adjoint_a=False,		# 如果为真, a则在进行乘法计算前进行共轭和转置
    				adjoint_b=False,		# 如果为真, b则在进行乘法计算前进行共轭和转置
    				a_is_sparse=False,		# 如果为真, a会被处理为稀疏矩阵
    				b_is_sparse=False,		# 如果为真, b会被处理为稀疏矩阵
    				name=None				# 操作的名字（可选参数）
					)
```

```python
import tensorflow.compat.v1 as tf
# 构建矩阵（二维张量）M x N
x = tf.constant([[1, 2], [3, 4]], tf.float32)
# 构建矩阵（二维张量）N x X
w = tf.constant([[-1], [-2]], tf.float32)
# 计算矩阵相乘
y = tf.matmul(x, w)
session = tf.Session()
# 打印结果
print("矩阵相乘结果：\n", session.run(y))
```

#### 4.3 张量的连接

Tensorflow通过函数concat实现张量的连接。

```python
tensorflow.concat(
    				values,			# 一个tensor的list或者tuple，里面是准备连接的矩阵或者数组。
    				axis,			# 准备连接的维度
    				name='concat'	# 操作的名字
					)
```

##### 4.3.1 二维张量连接

```python
import tensorflow.compat.v1 as tf
# 构建待拼接的二维张量t1
t1 = tf.constant([[1, 2, 3],
                  [4, 5, 6]], tf.float32)
# 构建待拼接的二维张量t2
t2 = tf.constant([[4, 5],
                  [7, 8]], tf.float32)
# 利用函数concat来实现对t1和t2的拼接
t = tf.concat([t1, t2], axis=1)
session = tf.Session()
# 打印结果
print("拼接后的张量：\n", session.run(t))
```

显然，如果张量沿着一个方向进行拼接，那么在另一个方向上的维数一定是相等的。

##### 4.3.2 三维张量连接

```python
import tensorflow.compat.v1 as tf
# 构建三维张量t1
t1 = tf.constant([[[9, 2], [1, 9]],
                  [[0, 9], [9, 9]]], tf.float32)
# 构建三维张量t2
t2 = tf.constant([[[5, 5], [5, 5]],
                  [[5, 5], [5, 5]]], tf.float32)
# 沿行方向“axis=0”进行连接
tt0 = tf.concat([t1, t2], axis=0)
# 沿列方向“axis=1”进行连接
tt1 = tf.concat([t1, t2], axis=1)
# 延深度方向“axis=2”进行连接
tt2 = tf.concat([t1, t2], axis=2)
session = tf.Session()
# 打印结果
print("三维张量t1:\n", session.run(t1))
print("三维张量t2:\n", session.run(t2))
print("沿‘0’方向连接：\n", session.run(tt0))
print("沿‘1’方向连接：\n", session.run(tt1))
print("沿‘2’方向连接：\n", session.run(tt2))
```

显然，如果两个三维张量在一个方向能够连接，那么这两个张量在另外两个方向上的维数是相等的。

#### 4.4 张量的堆叠

Tensorflow通过函数stack实现张量的堆叠，提升维度。

```python
tensorflow.stack(
    			  values,		# value:具有相同形状和类型的张量对象列表。
    			  axis=0,		# 要堆叠的轴，默认为第一个维度。负值环绕。
    			  name='stack'	# 此操作的名称(可选)
				)
```

##### 4.4.1 一维张量的堆叠

```python
import tensorflow.compat.v1 as tf
# 构造一维张量t1
t1 = tf.constant([1, 2, 3], tf.float32)
# 构造一维张量t2
t2 = tf.constant([4, 5, 6], tf.float32)
# 沿行方向‘axis=0’进行堆叠
tt0 = tf.stack([t1, t2], axis=0)
# 沿列方向‘axis=1’进行堆叠
tt1 = tf.stack([t1, t2], axis=1)
session = tf.Session()
# 打印结果
print("沿‘0’方向堆叠：\n", session.run(tt0))
print("沿‘1’方向堆叠：\n", session.run(tt1))
```

##### 4.4.2 二维张量的堆叠

```python
import tensorflow.compat.v1 as tf
# 构建二维张量t1
t1 = tf.constant([[11, 12, 13],
                  [14, 15, 16],
                  [17, 18, 19]], tf.float32)
# 构建二维张量t2
t2 = tf.constant([[4, 5, 6],
                  [7, 8, 9],
                  [1, 3, 2]], tf.float32)
# 沿‘0’方向进行堆叠
tt0 = tf.stack([t1, t2], axis=0)
# 沿‘1’方向进行堆叠
tt1 = tf.stack([t1, t2], axis=1)
# 沿‘2’方向进行堆叠
tt2 = tf.stack([t1, t2], axis=2)
session = tf.Session()
# 打印结果
print("沿‘0’方向进行堆叠:\n", session.run(tt0))
print("tt0的shape：", tt0.get_shape())
print("沿‘1’方向进行堆叠:\n", session.run(tt1))
print("tt1的shape：", tt0.get_shape())
print("沿‘2’方向进行堆叠:\n", session.run(tt2))
print("tt2的shape：", tt0.get_shape())
```

#### 4.5 张量的对比

Tensorflow通过函数equal实现张量对应值的对比。

```python
tensorflow.equal(x, y, name=None)	# 对张量x和y逐个元素进行比较，返回True或False
```

```python
import tensorflow.compat.v1 as tf
# 构建二维张量x
x = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], tf.float32)
# 构建二维张量y
y = tf.constant([[4, 5, 3],
                 [7, 5, 9],
                 [1, 3, 2]], tf.float32)
# 逐个元素进行对比
r = tf.equal(x, y)
session = tf.Session()
# 打印结果
print("张量x：\n", session.run(x))
print("张量y：\n", session.run(y))
print("对比结果：\n", session.run(r))
```

### 5. 占位符

占位符，可以理解为函数中的未知数。

假设有函数***f(x)=wx* **，其中***w***为3行2列的矩阵，则要进行函数矩阵相乘的运算，***x***需要满足必须有2行任意列。此时可以使用占位符表示***x***，再通过会话成员函数**run**中的参数**feed_dict**给占位符赋不同的值。

***placeholder()***函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict向占位符赋数据。

```python
tensorflow.placeholder(
    					dtype,			# 数据类型
   						shape=None,		# 数据形状，默认是None，就是一维值，也可以是多维
   						name=None		# 操作的名称
					   )
```

```python
import tensorflow.compat.v1 as tf
import numpy as np
# 创建3行2列二维张量w
w = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]], tf.float32)
# 占位符
x = tf.placeholder(tf.float32, [2, None], name='x')
# 进行矩阵相乘
y = tf.matmul(w, x)
session = tf.Session()
# 给占位符x赋值为2行2列的二维向量，并得到结果
result1 = session.run(y, feed_dict={x:np.array([[2, 1], [1, 2]], np.float32)})
# 给占位符x赋值为2行1列的二维向量，并得到结果
result2 = session.run(y, feed_dict={x:np.array([[-1],
                                                [2]], np.float32)})
# 打印结果
print("x为2行2列：\n", result1)
print("x为2行1列：\n", result2)
```

### 6. Variable对象

Tensorflow中的Variable类可以保存随时变化的值。

以下例子是创建一个Variable对象，将其初始化并且打印其值，然后利用成员函数assign_add改变其自身的值。

```python
import tensorflow.compat.v1 as tf
# 创建Variable对象x
v = tf.Variable(tf.constant([2, 3], tf.float32))
session = tf.Session()
# Variable对象初始化
session.run(tf.global_variables_initializer())
print("v初始化的值：\n", session.run(v))
# 使用成员函数assign_add改变本身的值
session.run(v.assign_add([10, 20]))
print("v的当前值：\n", session.run(v))
```

注：创建Variable对象后，要调用方法global_variables_initializer()，才可以使用Variable对象的值。

