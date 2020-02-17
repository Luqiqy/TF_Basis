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

