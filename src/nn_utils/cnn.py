import tensorflow as tf

def cnn_Kim(rep_tensor, filter_size, scope=None, keep_prob = 1., is_train=None,
           activation='relu', tensor_dict=None, name=''):
    output_conv = cnn(rep_tensor=rep_tensor, filter_size=filter_size, scope=scope, keep_prob=keep_prob, is_train=is_train,
                      activation=activation, tensor_dict=tensor_dict, name=name)

def cnn(rep_tensor, filter_size, scope = None, keep_prob = 1., is_train=None,
        activation='relu', tensor_dict = None, name=''):
    with tf.name_scope(scope or "convolution"):
        bs, sl, dim = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
        inputs = tf.expand_dims(rep_tensor, axis=-1)
        filter_shape = [filter_size, dim, 1, 1]
        w = tf.get_variable(name='conv_w', dtype = tf.float32,
                            initializer=tf.random_uniform(shape=filter_shape, minval=0., maxval=0.1))
        b = tf.get_variable(name='conv_b', dtype=tf.float32, initializer=tf.zeros([1]))
        conv = tf.nn.conv2d(inputs, filter=w, strides=[1,1,1,1], padding='VALID', name='conv')
        output = tf.nn.bias_add(conv, b)

        # activation
        output = tf.nn.relu(output)

        return output

def max_pooling(tensor, scope=None, name=''):
    with tf.name_scope(scope or "max_pooling"):
        return tf.reduce_max(tensor, axis=-1)
