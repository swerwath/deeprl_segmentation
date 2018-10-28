import tensorflow as tf
import numpy as np

#Performs a 2D Convolution operation with padding
def conv(input_layer, filter_size, kernel_size, name, strides, padding = "SAME", activation = tf.nn.relu):
    output = tf.layers.conv2d(input_layer, filters = filter_size, kernel_size = kernel_size, name = name, strides = strides, padding = padding, activation = activation)
    return output
    
def deconv(input_layer, filter_size, output_size, out_channel, in_channel, name, strides = [1, 1, 1], padding = "SAME", activation = tf.nn.relu):
    batch_size = tf.get_shape(input_layer)[0]
    output = tf.layers.conv2d_transpose(input_layer, filters = tf.get_variable(name = name, shape = [filter_size, filter_size, out_channel, in_channel]), kernel_size = tf.stack([batch_size, output_size, output_size, out_channel]), strides = strides, padding = padding, action = activation)
    return output

#Defines the unet structure, given a batch of input images of size 256 x 256
#Input: img_input is a tf Tensor of shape [batch_size, 256, 256]
def build_unet(img_input):
    res = img_input #256 x 256
    res = conv(res, 32 ,3, 'F0', strides = (2, 2)) #128 x 128
    res = conv(res, 64, 3, 'F1', strides = (2,2)) #64 x 64
    res = conv(res, 128, 3, 'F2', strides = (2,2)) #32 x 32
    res = conv(res, 256, 3, 'F3', strides = (2,2)) #16 x 16
    
    #Up-convolve
    #res = deconv(res, 1, 16, 256, 256, 'B0')
    res = deconv(res, 1, 32, 128, 256, 'B1', strides = [1, 2, 2])
    res = deconv(res, 1, 64, 64, 128, 'B2', strides = [1, 2, 2])
    res = deconv(res, 1, 128, 32, 64, 'B3', strides = [1, 2, 2])
    res = deconv(res, 1, 256, 16, 32, 'B4', strides = [1,2,2])
    return res               
