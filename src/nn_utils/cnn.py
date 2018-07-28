import tensorflow as tf
from configs import cfg

def cnn_Kim(rep_tensor, filter_height, out_channel_dims, scope=None, keep_prob = 1., is_train=None,
           activation='relu', tensor_dict=None, name=''):
    assert len(filter_height) == len(out_channel_dims)
    output = None
    for id,(fh,ocd) in enumerate(zip(filter_height, out_channel_dims)):
        scope_str = 'filter_height_%d' % fh
        with tf.variable_scope(scope_str):
            output_conv = cnn(rep_tensor=rep_tensor, filter_size=fh, out_channel_dim=ocd, scope=None, keep_prob=keep_prob, is_train=is_train,
                      activation=activation, tensor_dict=tensor_dict, name=name)
            output_pooling = max_pooling(output_conv, 1, None, name)
            if id == 0:
                output = output_pooling
            else:
                output = tf.concat([output, output_pooling], axis=-1)
    return output

def cnn(rep_tensor, filter_size, out_channel_dim, scope = None, keep_prob = 1., is_train=None,
        activation='relu', tensor_dict = None, name=''):
    with tf.name_scope(scope or "convolution"):
        bs, sl, dim = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
        inputs = tf.expand_dims(rep_tensor, axis=-1)
        filter_shape = [filter_size, cfg.word_embedding_length, 1, out_channel_dim]
        w = tf.get_variable(name='conv_w', dtype = tf.float32,
                            initializer=tf.truncated_normal(filter_shape, stddev=0.1))
        b = tf.get_variable(name='conv_b', dtype=tf.float32, initializer=tf.constant(0.1, shape = [out_channel_dim]))
        conv = tf.nn.conv2d(inputs, filter=w, strides=[1,1,1,1], padding='VALID', name='conv')
        output = tf.nn.bias_add(conv, b)

        # activation
        output = tf.nn.relu(output)

        # reshape
        output = tf.squeeze(output, [2])

        return output

def max_pooling(tensor, axis, scope=None, name=''):
    with tf.name_scope(scope or "max_pooling"):
        return tf.reduce_max(tensor, axis=axis)
