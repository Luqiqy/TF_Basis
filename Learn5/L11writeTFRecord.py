import tensorflow.compat.v1 as tf
import numpy as np
# 创建文件
record = tf.python_io.TFRecordWriter('data.tfrecord')
# 3个ndarray
array1 = np.array([[[1, 2, 1, 2], [3, 4, 2, 9], [5, 6, 0, 3]],
                   [[7, 8, 1, 6], [9, 6, 1, 7], [1, 2, 5, 9]]], np.float32)
array2 = np.array([[[11, 12, 11], [13, 14, 12], [15, 16, 13]],
                   [[17, 18, 11], [19, 10, 11], [11, 12, 15]],
                   [[13, 14, 15], [18, 11, 12], [19, 14, 10]]], np.float32)
array3 = np.array([[[21, 23, 21], [23, 24, 22]],
                   [[27, 28, 24], [29, 20, 21]]], np.float32)
# 将3个array存在列表arrays中方便遍历
arrays = [array1, array2, array3]
# 循环处理上述列表中的每一个元素
for array in arrays:
    # 计算得每一个ndarray的形状（高，宽，深度）
    height, width, depth = array.shape
    # 将ndarray中的值转换为字节类型
    array_raw = array.tostring()
    # ndarray的值及对应的高、宽、深度
    feature = {
        'array_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_raw])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth]))
                }
    # 用于构造每个样本的信息键值对
    features = tf.train.Features(feature=feature)
    # 用于写入TFRecord文件，返回example协议格式块
    example = tf.train.Example(features=features)
    # 字符串序列化后写入文件
    record.write(example.SerializeToString())
record.close()