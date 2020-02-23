import tensorflow.compat.v1 as tf
import numpy as np
record = tf.python_io.TFRecordWriter('dataTest.tfrecord')
array1 = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
array2 = np.array([[[11, 12, 13], [14, 15, 16]]], np.float32)
array3 = np.array([[[21, 22, 23], [23, 24, 25]]], np.float32)
arrays = [array1, array2, array3]
for array in arrays:
    array_raw = array.tostring()
    feature = {'array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_raw]))}
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    record.write(example.SerializeToString())
record.close()