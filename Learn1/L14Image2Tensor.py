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
