## 四. 全连接神经网络

神经网路可以理解为一种特殊的较为复杂的向量变换。全连接神经网络就是一种复杂的变换规则。

全连接神经网络一般用图的形式分层来表示。最左边的一层为**输入层**，中间各层为**隐含层**，最右边一层为**输出层**，从第0层由左到右开始编号。每一层都有若干节点，这些节点在神经网络中被称为**神经元**。每一层神经元和下一层神经元之间会通过一条弧相连接，弧上都有一个值，常称为**权重**或**权值**，常用符号***w***表示，弧上还有一个公共值，常称为**偏置**，常用符号***b***表示，也可以用矩阵的形式管理每一层的权重和偏置。

### 1. 神经网络的矩阵表达

利用矩阵的形式计算神经网络：

* 线性组合，加权求和及偏置
* 将以上线性组合作为激活函数的输入，即可得下一层神经元的值

#### 1.1 输入数据按列存储

> 输入层输入数据（列向量）：[3, 5]^T^

> 权重w1（2x3矩阵）:	[[1, 2, 7],
>
> ​										[ 2, 6, 8]]

> 偏置b1（3*1矩阵）:	[[-4], [2], [1]]

> 激活函数：	f = 2x

> 权重w2（3x2矩阵）:	[[2, 3],
>
> ​										[1, -2],
>
> ​										[-1, 1]]

> 偏置b2（2x1矩阵）:	[[5], [-3]]

> 激活函数：	f = 2x

```python
import tensorflow.compat.v1 as tf
import numpy as np
# 输入层
x = tf.placeholder(tf.float32, (2, None))
# 第一层的权重矩阵
w1 = tf.constant([[1, 4, 7],
                  [2, 6, 8]], tf.float32)
# 第一层的偏置
b1 = tf.constant([[-4], [2], [1]], tf.float32)
# 计算第一层的线性组合，w1转置后与x矩阵相乘
l1 = tf.matmul(w1, x, True)+b1
# 激活函数f=2x
sigma1 = 2*l1
# 第二层的权重矩阵
w2 = tf.constant([[ 2,  3],
                  [ 1, -2],
                  [-1,  1]], tf.float32)
# 第二层的偏置
b2 = tf.constant([[5], [-3]], tf.float32)
# 计算第二层的线性组合，w1转置后与sigma1相乘
l2 = tf.matmul(w2, sigma1, True)+b2
# 激活函数f=2x
sigma2 = 2*l2
# 创建会话
sess = tf.Session()
# 令输入数据x=[[3], [5]]
print(sess.run(sigma2, {x: np.array([[3], [5]], np.float32)}))
```

#### 1.2 输入数据按行存储

> 输入层输入数据（4x2矩阵）：	[[10, 11],
>
> ​														  [20, 21],
>
> ​														  [30, 31],
>
> ​														  [40, 41]]

> 权重w1（2x3矩阵）:	[[1, 2, 7],
>
> ​										[ 2, 6, 8]]

> 偏置b1（3*1矩阵）:	[1, 4, 7]

> 激活函数：	f = 2x

> 权重w2（3x2矩阵）:	[[2, 3],
>
> ​										[1, -2],
>
> ​										[-1, 1]]

> 偏置b2（2x1矩阵）:	[[5], [-3]]

> 激活函数：	f = 2x

```python
import tensorflow.compat.v1 as tf
import numpy as np
# 输入层, 每一个输入按行存储
x = tf.placeholder(tf.float32, (None, 2))
# 第一层的权重矩阵
w1 = tf.constant([[1, 4, 7],
                  [2, 6, 8]], tf.float32)
# 第一层的偏置
b1 = tf.constant([1, 4, 7], tf.float32)
# b1 = tf.constant([[1, 4, 7]], tf.float32)
# 计算第一层的线性组合，w1转置后与x矩阵相乘
l1 = tf.matmul(x, w1)+b1
# 激活函数f=2x
sigma1 = 2*l1
# 第二层的权重矩阵
w2 = tf.constant([[ 2,  3],
                  [ 1, -2],
                  [-1,  1]], tf.float32)
# 第二层的偏置
b2 = tf.constant([5, -3], tf.float32)
# 计算第二层的线性组合，w1转置后与sigma1相乘
l2 = tf.matmul(sigma1, w2)+b2
# 激活函数f=2x
sigma2 = 2*l2
# 创建会话
sess = tf.Session()
# 输入数据为4x2矩阵x，四个输入
print(sess.run(sigma2, {x: np.array([[10, 11],
                                     [20, 21],
                                     [30, 31],
                                     [40, 41]], np.float32)}))
```

#### 1.3 区别

权重矩阵行数等于上层神经元的个数，列数为下层神经元的个数。

* 输入数据按列存储，在利用矩阵相乘的函数tf.matmul时，需要对权重矩阵进行转置，偏置也要存储在一个二维张量中。
* 多个输入，每一个输入按行存储，偏置声明为一个一维张量。

### 2. 激活函数

#### 2.1 sigmoid激活函数

sigmoid激活函数的形式：
$$
sigmoid(x) =\frac{1}{1+e^{-x}}
$$
sigmoid函数求导后形式：
$$
sigmoid'(x)=\frac{e^{-x}}{\left ( 1+e^{-x} \right )}
$$
范围：	sigmoid ∈ (0, 1)，sigmoid ∈ (0, 0.25) 

关系：
$$
sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))
$$
Tensorflow通过函数`tf.nn.sigmoid(x, name=None)`实现sigmoid函数

```python
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# x的取值
x_value = np.arange(-10, 10, 0.01)
sess = tf.Session()
# x为自变量通过sigmoid函数后得到y_value
y_value = sess.run(tf.nn.sigmoid(x_value))
# 对sigmoid求导f'(x) = f(x)*(1-f(x)) 
Y_value = y_value*(1-y_value)
# 画sigmoid图
plt.figure(1)
plt.plot(x_value, y_value)
plt.show()
# 画sigmoid导数的图
plt.figure(2)
plt.plot(x_value, Y_value)
plt.show()
```

#### 2.2 tanh激活函数

tanh激活函数的形式：
$$
tanh(x) =\frac{1-e^{-2x}}{1+e^{-2x}}=\frac{2}{1+e^{-2x}}-1
$$
tanh函数求导后形式：
$$
tanh'(x) =\frac{4e^{-2x}}{\left(1+e^{-2x}\right)^{2}}=1-(tanh(x))^{2}
$$
范围：	tanh(x) ∈ (-1, 1)，tanh'(x) ∈ (0, 1]

关系：
$$
tanh'(x) = 1-(tanh(x))^{2}
$$
Tensorflow通过函数`tf.nn.tanh(x, name=None)`实现tanh函数

```python
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# x的取值
x_value = np.arange(-10, 10, 0.01)
sess = tf.Session()
# x为自变量通过tanh函数后得到y_value
y_value = sess.run(tf.nn.tanh(x_value))
# 对tanh求导
Y_value = 1-np.power(y_value, 2.0)
# 画tanh图
plt.figure(1)
plt.plot(x_value, y_value)
plt.show()
# 画tanh导数的图
plt.figure(2)
plt.plot(x_value, Y_value)
plt.show()
```

#### 2.3 Relu激活函数

Relu激活函数的形式：
$$
relu(x)=max(x,0)=\left\{\begin{matrix}
x, \:\:\:\:x\geq 0\\
0, \:\:\:\:x<0\\ 

\end{matrix}\right.
$$
Relu函数求导后形式：
$$
relu’(x)=\left\{\begin{matrix}
1, \:\:\:\:x\geq 0\\
0, \:\:\:\:x<0\\ 

\end{matrix}\right.
$$
范围：	relu(x) ∈ [0, +∞]，relu(x) ∈ {0, 1}

Tensorflow通过函数`tf.nn.relu(features, name=None)`实现relu激活函数

***例***：有三元函数***g(x~1~, x~2~, x~3~)=2x~1~+3x~2~+4x~3~***，函数***f(x~1~, x~2~, x~3~)=relu(g)=relu(2x~1~+3x~2~+4x~3~)***，计算f在(2, 1, 3)处的导数。

```python
import tensorflow.compat.v1 as tf
# 变量x
x = tf.Variable(tf.constant([[2, 1, 3]], tf.float32))
#
w = tf.constant([[2], [3], [4]], tf.float32)
g = tf.matmul(x, w)
f = tf.nn.relu(g)
gradient = tf.gradients(f, [g, x])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(gradient))
```

#### 2.4 leaky relu激活函数

leaky relu激活函数的形式：
$$
leakyrelu(x)=\left\{\begin{matrix}
x, \:\:\:\:\:\:\:x\geq 0\\
\alpha x, \:\:\:\:x<0\\ 

\end{matrix}\right.
$$
leaky relu函数求导后形式：
$$
leakyrelu'(x)=\left\{\begin{matrix}
1, \:\:\:\:\:\:x\geq 0\\
\alpha, \:\:\:\:\:x<0\\ 

\end{matrix}\right.
$$
范围：	leakyrelu(x) ∈ (-∞, +∞)，leakyrelu'(x) ∈ {1, α}

Tensorflow通过函数`tf.nn.leaky_relu(features, alpha=0.2, name=None)`实现leaky relu激活函数

```python
import tensorflow.compat.v1 as tf
# 变量x
x = tf.Variable(tf.constant([[2, 1, 3]], tf.float32))
w = tf.constant([[2], [3], [4]], tf.float32)
# 函数g
g = tf.matmul(x, w)
# 函数f=leaky_relu(g)
f = tf.nn.leaky_relu(g, alpha=0.2)
# 牛顿梯度下降法
opti = tf.train.GradientDescentOptimizer(0.5).minimize(f)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 打印结果
for i in range(5):
    sess.run(opti)
    print('第%d次迭代的值'%(i+1))
    print(sess.run(x))
```

#### 2.5 elu激活函数

elu激活函数的形式：
$$
elu(x)=\left\{\begin{matrix}x, \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:x\geq 0\\\alpha (e^{x}-1), \:\:\:\:x<0\\ \end{matrix}\right.
$$
elu函数求导后形式：
$$
elu'(x)=\left\{\begin{matrix}1, \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\,\,\,\,\:\:\:\:\:\:\:\:\:\:\:x\geq 0\\\alpha e^{x}=elu(x)+α, \:\:\:\:\:x<0\\ \end{matrix}\right.
$$
范围：	当α=1时，elu(x) ∈ (-1, +∞)，elu'(x) ∈ (0, 1]

Tensorflow通过函数`tf.nn.elu(features, name=None)`实现α=1时的elu激活函数

```python
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
# x的取值
x_value = np.arange(-10, 10, 0.01)
sess = tf.Session()
# x为自变量通过elu函数后得到y_value
y_value = sess.run(tf.nn.elu(x_value))
# 画elu图
plt.figure(1)
plt.plot(x_value, y_value)
plt.show()
```

#### 2.6 crelu激活函数

crelu激活函数的形式：
$$
crelu(x)=(max(x,0), \:\: max(-x,0))
$$
***例***：input x=-2，crelu(-2)=(max(-2, 0), max(2, 0))=(0, 2)

​		input x=[2,-1]，crelu([2, -1])=(max(2, 0), max(-2, 0), max(-1, 0), max(1, 0)) = (2, 0, 0, 1)

Tensorflow通过函数`tf.nn.crelu(features, name=None)`实现crelu激活函数

#### 2.7 selu激活函数

selu激活函数的形式：
$$
selu(x)=\left\{\begin{matrix}
\lambda\alpha(e^{x}-1), \:\:\:\:\:x<0\\ 
\lambda x, \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:x\geq0\\ 
\end{matrix}\right.
$$
selu函数求导后形式：
$$
selu'(x)=\left\{\begin{matrix} 
\lambda \alpha e^{x}, \:\:\: x<0\\ 
\lambda, \:\:\:\:\:\:\:\:\:\:x \geq 0
\end{matrix}\right.
$$
Tensorflow通过函数`tf.nn.selu(features, name=None)`实现selu函数

#### 2.8 relu6激活函数

relu6激活函数的形式：
$$
relu6(x)=min(max(x,0),6)
$$
relu6函数求导后形式：
$$
relu6'(x)=\left\{ 
\begin{matrix}
1, \:0<x<6 \\
0, \:\:\:\:\:\:\:\:\:\:\:\:\:其他

\end{matrix}
\right.
$$
范围：	relu6(x) ∈ [0, 6]，relu6'(x) ∈ {0, 1}

Tensorflow通过函数`tf.nn.relu6(features, name=None)`实现relu6函数

#### 2.9 softplus激活函数

softplus激活函数的形式：
$$
softplus(x)=ln(1+e^{x})
$$
softplus函数求导后形式：
$$
softplus'(x)=\frac{e^{x}}{1+e^{x}}
$$
范围：	softplus(x) ∈ (0, +∞)， softplus'(x) ∈ (0, 1)

Tensorflow通过函数`tf.nn.softplus(features, name=None)`实现softplus函数

#### 2.10 softsign激活函数

softsign激活函数的形式：
$$
softsign(x)=\frac{x}{1+|x|}
$$
softsign函数求导后形式：
$$
softsign'(x)=\frac{|x|}{1+ |x|^{2}}
$$
范围：	softsign(x) ∈ (-1, 1)， softsign'(x) ∈ (0, 1)

Tensorflow通过函数`tf.nn.softsign(features, name=None)`实现softsign函数