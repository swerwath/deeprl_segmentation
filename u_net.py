import tensorflow as tf
import numpy as np

#Performs a 2D Convolution operation with padding
def conv(input_layer, filter_size, kernel_size, name, strides, padding = "SAME", activation = tf.nn.relu):
    output = tf.layers.conv2d(input_layer, filters = filter_size, kernel_size = kernel_size, name = name, strides = strides, padding = padding, activation = activation)
    return output
    
def deconv(input_layer, filter_size, output_size, out_channel, in_channel, name, strides = [1, 1], padding = "SAME"):
    batch_size = tf.shape(input_layer)[0]
    output = tf.nn.conv2d_transpose(input_layer, tf.get_variable(name = name, shape = [filter_size, filter_size, out_channel, in_channel]), tf.stack([batch_size, output_size, output_size, out_channel]), strides = strides, padding = padding)
    return output

#Defines the unet structure, given a batch of input images of size 256 x 256
#Input: img_input is a tf Tensor of shape [batch_size, 256, 256, 1]. img_input should have type 'float32'. 
#Optional Args: scope name, reuse (added for compatibility purposes)
#Output: Returns a tensor of shape [batch_size, 256, 256, 1] representing the q-values. 
def build_unet(img_input, scope = "default", reuse = False):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(img_input)[0]
        res = img_input #256 x 256
        res = conv(res, 32 ,3, 'F0', strides = (2, 2)) #128 x 128
        res = conv(res, 64, 3, 'F1', strides = (2,2)) #64 x 64
        res = conv(res, 128, 3, 'F2', strides = (2,2)) #32 x 32
        res = conv(res, 256, 3, 'F3', strides = (2,2)) #16 x 16
        
        #2 FC layers to get the convolved tensor down to 3 values for pen-state
        pen_states = tf.contrib.layers.flatten(res)
        pen_states = tf.contrib.layers.fully_connected(pen_states, 300)
        pen_states = tf.contrib.layers.fully_connected(pen_states, 3)

        #Up-convolve
        res = deconv(res, 3, 16, 256, 256, 'B0') #16 x 16
        res = tf.nn.relu(res)

        res = deconv(res, 3, 32, 128, 256, 'B1', strides = (2, 2)) #32 x 32
        res = tf.nn.relu(res)

        res = deconv(res, 3, 64, 64, 128, 'B2', strides = (2,2)) #64 x 64
        res = tf.nn.relu(res)

        res = deconv(res, 3, 128, 32, 64, 'B3', strides = (2,2)) #128 x 128
        res = tf.nn.relu(res)

        res = deconv(res, 3, 256, 1, 32, 'B4', strides = (2,2)) #256 x 256
        res = tf.squeeze(tf.nn.relu(res))
        return res, pen_states 

#Example usage
def main():
    img = tf.convert_to_tensor(np.random.uniform(0, 1, size = (10, 256, 256, 6)).astype('float32'))
    ans, pen = build_unet(img)
    print(ans.get_shape(), pen.get_shape())

if __name__ == "__main__":
    main()
