## 五. 神经网络处理分类问题

### 待理解补充……1. TFRecord

TFRecord是Tensorflow设计的一种存储数据的内置文件格式，可以方便高效地管理数据及这些数据的相关信息，利用它可以将数据快速加载到内存中。

#### 1.1 将ndarray写入TFRecord文件

***例：***设由3个三维的ndarray，尺寸依次是2行3列4深度、3行3列3深度和2行2列3深度，将这三个ndarray及其对应的尺寸写入到data.tfrecord的TFRecord文件中。

**准备工作及思考：**

* 建立TFRecord存储器

  ```python
  tf.python_io.TFRecordWriter(path)  
  #写入tfrecord文件
  #path为tfrecord的存储路径
  ```

* 构造每个样本的example模块

  ```python
  # Example协议块的规则如下
  message Example {
    Features features = 1;
  };
  message Features {
    map<string, Feature> feature = 1;
  };
  message Feature {
    oneof kind {
      BytesList bytes_list = 1;
      FloatList float_list = 2;
      Int64List int64_list = 3;
    }
  };
  ```



* ```python
  tf.train.Example(features = None)  
  #用于写入tfrecords文件
  #features ： tf.train.Features类型的特征实例
  #返回example协议格式块
  tf.train.Features(feature = None)
  #用于构造每个样本的信息键值对
  #feature : 字典数据，key为要保存的名字，value为tf.train.Feature实例
  #返回Features类型
  tf.train.Feature(**options) 
  #options可选的三种数据格式：
  bytes_list = tf.train.BytesList(value = [Bytes])
  int64_list = tf.train.Int64List(value = [Value])
  float_list = tf.trian.FloatList(value = [Value])
  ```

* 

tf_example可以写入的数据形式有三种，分别是*BytesList*, *FloatList*以及*Int64List*的类型。

（1）`tf.train.Example(features = None)` 这里的features是tf.train.Features类型的特征实例。
 （2）`tf.train.Features(feature = None)` 这里的feature是以字典的形式存在，*key：要保存数据的名字    value：要保存的数据，但是格式必须符合tf.train.Feature实例要求。



### 待补充……2. 建立分类问题的数学模型

#### 2.1 数据类别（标签）

Tensorflow通过函数`one_hot( )`实现类别的数字化。

```python
tensorflow.one_hot(
    	indices,		# 输入为张量
    	depth,			# 定义one_hot深度的标量
    	on_value=None,	# 
    	off_value=None, # 
    	axis=None,		# 要填充的轴（默认值：-1）
    	dtype=None,		# 输出张量的数据类型。
    	name=None		# 操作的名称
)						# 输出：The one-hot tensor.
```

```python
import tensorflow.compat.v1 as tf
# depth=10代表有10个类别，因此返回张量有10列，axis=1代表按照每一行存储类别的数字化
v = tf.one_hot([0, 1, 9, 2, 8, 3, 4, 7, 6, 5], depth=10, axis=1, dtype=tf.float32)
session = tf.Session()
print(session.run(v))
```

2.2 图形与TFRecord

### 3. 损失函数与训练模型

#### 3.1 sigmoid损失函数

假设***y***（labels）为人工分类的标签，***_y***（logits）代表全连接神经网络的输出层的值，两者的交叉熵定义为：
$$
-y *ln(sigmoid(\_y)))-(1-y)*ln(1-sigmoid(\_y))\\
=-y*ln(\frac{1}{1+e^{-\_y}})-(1-y)*ln(\frac{e^{-\_y}}{1+e^{-\_y}})=\_y-\_y*y+ln(1+e^{-\_y})
$$
Tensorflow通过函数`sigmoid_cross_entropy_with_logits()`函数实现sigmoid交叉熵。

```python
tensorflow.nn.sigmoid_cross_entropy_with_logits(
    _sentinel=None,		# 
    labels=None,		# float32或float64型张量，同logits
    logits=None,		# float32或float64型张量
    name=None			# 操作的名称
)
```

```python
import tensorflow.compat.v1 as tf
# 输出层的值
logits = tf.constant([[-8.29, 0.64, 9.22, -0.08, 5.6, 6.15, -10.21, 7.51, 7.73, 9.43]],
                     tf.float32)
# 人工分类的标签
labels = tf.constant([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], tf.float32)
# sigmoid交叉熵
entroy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
# 损失值
loss = tf.reduce_sum(entroy)
# 打印损失值
session = tf.Session()
print(session.run(loss))
```

labels代表人工分类标签，是一个N行C列的二维张量，代表N个样本的标签吗，每一行代表一个样本的分类标签。logits代表输出层的结果，与labels的尺寸相同，每一行代表一个样本经过全连接神经网络后的输出值，利用求和函数reduce_sum对函数sigmoid_cross_entropy_with_logits返回的结果求和，其结果为sigmoid交叉熵损失函数。

#### 3.2 softmax损失函数

##### 3.2.1 softmax计算原理

假设向量***x***=(x~1~, x~2~, … , x~m~)，对***x***进行softmax处理方式如下：
$$
softmax(x)=(\frac{e^{x_1}}{\sum_{i=1}^{m}e^{x_i}}, \frac{e^{x_2}}{\sum_{i=1}^{m}e^{x_i}}, …,\frac{e^{x_1}}{\sum_{i=m}^{m}e^{x_i}})
$$
显然，***x***进行softmax处理后归一化为[0, 1]，且和为1。

Tensorflow通过函数`softmax( )`实现softmax处理。

```python
tf.nn.softmax(
    logits,		# half,float32或float64类型的张量
    axis=None,	# 进行softmax的维度
    name=None	# 操作的名称
)				# 返回与logits类型和shape相同的张量
```

```python
import tensorflow.compat.v1 as tf
# 输入张量
x = tf.constant([[2, 1, 2],
                 [2, 2, 2]], tf.float32)
# 分别对每一行（axis=1）进行softmax处理
s = tf.nn.softmax(x, axis=1)
# 打印结果
session = tf.Session()
print(session.run(s))
```

##### 3.2.2 softmax熵及softmax损失函数

***y***（labels）为人工分类的标签，***_y***（logits）代表全连接神经网络的输出层的值

***y***和***_y***的softmax熵的定义为：
$$
-y*log(softmax(\_y))
$$

* 利用Tensorflow中的求和函数reduce_sum和函数tf.nn.softmax定义softmax损失函数为tf.reduce_sum(-y*tf.nn.softmax(_y,1))

  ```python
  import tensorflow.compat.v1 as tf
  # 假设_y为全连接神经网络的输出（输出层有3个神经元）
  _y = tf.constant([[0, 2, -3], [4, -5, 6]], tf.float32)
  # 人工分类结果
  y = tf.constant([[1, 0, 0], [0, 0, 1]], tf.float32)
  # 计算softmax熵
  _y_softmax = tf.nn.softmax(_y)
  entropy = tf.reduce_sum(-y*tf.log(_y_softmax), 1)
  # loss损失函数
  loss = tf.reduce_sum(entropy)
  # 打印结果
  session = tf.Session()
  print(session.run(loss))
  ```

* Tensorflow通过softmax_cross_entropy_with_logits_v2可以直接得到softmax熵

  ```python
  import tensorflow.compat.v1 as tf
  # 假设_y为全连接神经网络的输出（输出层有3个神经元）
  _y = tf.constant([[0, 2, -3], [4, -5, 6]], tf.float32)
  # 人工分类结果
  y = tf.constant([[1, 0, 0], [0, 0, 1]], tf.float32)
  # 计算softmax熵
  entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_y, labels=y)
  # loss损失函数
  loss = tf.reduce_sum(entropy)
  # 打印结果
  session = tf.Session()
  print(session.run(loss))
  ```

  

