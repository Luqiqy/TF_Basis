## 五. 神经网络处理分类问题

### 1. TFRecord

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