## 二. 梯度及梯度下降法

### 1. 梯度

Tensorflow通过函数`gradients()`实现自动计算梯度。

```python
tf.gradients(			# 实现ys对xs求导
     ys, xs,     		# 可以是tensor，也可以是list，形如[tensor1, tensor2, …, tensorn]
     grad_ys=None,		# grad_ys是个长度等于len(ys)的list，可对xs中的每个元素的求导加权重。
     name='gradients',	# 操作的名称
     colocate_gradients_with_ops=False,
     gate_gradients=False,
     aggregation_method=None,
     stop_gradients=None)
```

```python
import tensorflow.compat.v1 as tf
import numpy as np
# 通过矩阵乘法构建函数F=(3x1+4x2)**2
w = tf.constant([[3, 4]], tf.float32)
x = tf.placeholder(tf.float32, (2, 1))
y = tf.matmul(w, x)
F = tf.pow(y, 2)
# 开始计算函数F的梯度
grads = tf.gradients(F, x)
session = tf.Session()
# 打印结果
print(session.run(grads, {x: np.array([[2], [3]])}))
'''
>>[array([[108.],
          [144.]], dtype=float32)]
'''
```

*****

### 2. 求导链式法则

#### 2.1 多个函数和的导数

多个函数和的导数等于多个函数导数的和。

#### 2.2 复合函数的导数

层层求导

#### 2.3 单变量函数的驻点、极值点和鞍点

##### 2.3.1 驻点

驻点是指函数的一阶导数为0的点

##### 2.3.2 极值点

* 极小值是某点的值小于或等于在该点附近的任何点的函数值

  可导函数在某点处为极小值，则该函数在该点处一阶导数等于0（为驻点）且二阶导数大于0

* 极大值是某点的值大于或等于在该点附近的任何点的函数值

  可导函数在某点处为极大值，则该函数在该点处一阶导数等于0（为驻点）且二阶导数小于0

##### 2.3.3 鞍点

鞍点是函数在该点的一阶导数等于0（为驻点）且二阶导数也等于0的点

鞍点是不是极值点，要视情况而定

#### 2.4 多变量函数的驻点、极值点和鞍点

##### 2.4.1 驻点

驻点是多变量函数关于自变量的梯度等于0向量的点

##### 2.4.2 极值点

* 极小值是可导函数在该点为驻点，且在该点的**Hessian**矩阵的特征值全大于0
* 极大值是可导函数在该点为驻点，且在该点的**Hessian**矩阵的特征值全小于0

##### 2.4.3 鞍点

鞍点是可导函数在该点为驻点，且在该点的**Hessian**矩阵的特征值有的大于0，有的小于0

鞍点有可能是极大值点或者极小值点。以极小值为例，一个函数可能有多个极小值，被称为局部极小值，其中最小的被称为最小值或全局最小值，对应的点也被称为最小值点或全局极小值点。

##### 2.4.4 凸函数

凸函数的极小值点和最小值点是同一个点，其局部极小值点为全局最小值点。

#### 2.5 函数的泰勒级数展开

2.5.1 一元函数的泰勒级数展开

2.5.2 多元函数的泰勒级数展开

#### 2.6 梯度下降法

```python
Tf.train.GradientDescentOptimizer(
    learning_rate, 			# 学习率，控制参数的更新速度
    use_locking=False,		# 
	name='GradientDescent	# 操作的名称
	)
```

##### 2.6.1 一元函数的梯度下降法

##### 2.6.2 多元函数的梯度下降法

*****

### 3.梯度下降法

针对标准的梯度下降法，有很多改进方法。Tensorflow已经实现了一些经典的常用方法。

#### 3.1 Adagrad法

Tensorflow通过函数`tensorflow.train.AdagradOptimizer()`实现Adagrad梯度下降。

```python
tensorflow.train.AdagradOptimizer(
    learning_rate, 					# 学习率，控制参数的更新速度
    initial_accumulator_value=0.1,  # 类型为浮点值。累加器的起始值必须为正。
    use_locking=False, 				# True为锁
    name='Adagrad')					# 操作的名称
```

```python
import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# Adagrad梯度下降
opti = tf.train.AdagradOptimizer(0.25, 0.1).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))
'''  
>>[[3.750195 ]
   [2.7500868]]
  [[3.579276 ]
   [2.5811846]]
  [[3.442659 ]
   [2.4473038]]
'''
```

#### 3.2 Momentum法

Tensorflow通过函数`tensorflow.train.MomentumOptimizer()`实现Momentum梯度下降。

```python
tf.train.MomentumOptimizer(
    learning_rate,			# 学习率，控制参数的更新速度
    momentum,				# 动量，一个张量或一个浮点值
    use_locking=False,		# True为锁
    name='Momentum')        # 操作的名称
```

```python
import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# Momentum梯度下降
opti = tf.train.MomentumOptimizer(0.01, 0.9).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))
'''
>>[[3.92]
   [2.88]]
  [[3.7696002]
   [2.6568   ]]
  [[3.5588481]
   [2.349648 ]]
'''
```

#### 3.3 NAG法

NAG法是对Momentum的改进。Tensorflow通过函数MonmentumOptimizer实现Monmentum法，该函数中有一个参数use_nesterov，只要将该参数设置为True，就是Tensorflow实现的NAG法。

```python
import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# NAG梯度下降
opti = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))
'''
[[3.848]
 [2.772]]
[[3.636976]
 [2.464128]]
[[3.3781133]
 [2.0995615]]
'''
```

#### 3.4 RMSprop法

Tensorflow通过函数`tensorflow.train.RMSPropOptimizer()`实现RMSProp梯度下降。

```python
tf.train.RMSPropOptimizer(
    learning_rate, 			# 学习率，控制参数的更新速度
    decay, 					# discounting factor for the history/coming gradient
    momentum=0.0, 			# a scalar tensor
    epsilon=1e-10, 			# 防止分母趋近于0
    use_locking=False, 		# True为锁
    name='RMSProp') 		# 操作的名字
```

```python
import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# RMSProp梯度下降
opti = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, epsilon=1e-10).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))
'''
[[3.9703906]
 [2.9693215]]
[[3.9482605]
 [2.946826 ]]
[[3.9295564]
 [2.927947 ]]
 '''
```

#### 3.5 具备动量的RMSprop法

具备动量的RMSprop法是RMSprop和Momentum结合的方法，在`tensorflow.train.RMSPropOptimizer()`函数中设置参数momentum。

```python
import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# 具备动量的RMSProp梯度下降
opti = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, momentum=0.9, epsilon=1e-10).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))
'''
[[3.9703906]
 [2.9693215]]
[[3.9216123]
 [2.9192154]]
[[3.85909  ]
 [2.8553555]]
'''
```

#### 3.6 Adadelta法

Tensorflow通过函数`tensorflow.train.AdadeltaOptimizer()`来实现Adadelta梯度下降。

```python
tf.train.AdadeltaOptimizer(
    learning_rate, 			# 学习率，控制参数的更新速度
    rho=0.95, 				# 衰减率，一个张量或一个浮点值
    epsilon=1e-10, 			# 防止分母趋近于0
    use_locking=False, 		# True为锁
    name='Adadelta') 		# 操作的名字
```

```python 
import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# Adadelta梯度下降
opti = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-8).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))
'''
[[3.9999995]
 [2.9999995]]
[[3.999999]
 [2.999999]]
[[3.9999986]
 [2.9999986]]
'''
```

#### 3.7 Adam法

Tensorflow通过函数`tensorflow.train.AdamOptimizer()`来实现Adadelta梯度下降。

```python
tf.train.AdamOptimizer(
    learning_rate, 			# 学习率，控制参数的更新速度
    beta1=0.9,				# 浮点值或恒定浮点张量，第一次矩估计的指数衰减率
    beta2=0.999,		    # 浮点值或恒定浮点张量，第一次矩估计的指数衰减率
    epsilon=1e-10, 			# 防止分母趋近于0
    use_locking=False, 		# True为锁
    name='Adadelta') 		# 操作的名字
```

```python
import tensorflow.compat.v1 as tf
# 初始化变量x的值
x = tf.Variable(tf.constant([[4], [3]], tf.float32), dtype=tf.float32)
w = tf.constant([[1, 2]], tf.float32)
y = tf.reduce_sum(tf.matmul(w, tf.square(x)))
# Adam梯度下降
opti = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(y)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# 打印迭代结果
for i in range(3):
    session.run(opti)
    print(session.run(x))
'''
[[3.999]
 [2.999]]
[[3.9980001]
 [2.9980001]]
[[3.9970002]
 [2.9970002]]
'''
```

#### 3.8 Batch梯度下降

#### 3.9 随机梯度下降

#### 3.10 mini-Batch梯度下降

