import tensorflow as tf
from functools import reduce
from operator import mul

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out