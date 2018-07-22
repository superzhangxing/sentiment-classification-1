import tensorflow as tf

# -------------- emb mat--------------
def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_mat=None,
                           extra_trainable=False, scope=None):
    """
    generate embedding matrix for looking up
    :param dict_size: indices 0 and 1 corresponding to empty and unknown token
    :param emb_len:
    :param init_mat: init mat matching for [dict_size, emb_len]
    :param extra_mat: extra tensor [extra_dict_size, emb_len]
    :param extra_trainable:
    :param scope:
    :return: if extra_mat is None, return[dict_size+extra_dict_size,emb_len], else [dict_size,emb_len]
    """
    with tf.variable_scope(scope or 'gene_emb_mat'):
        emb_mat_ept_and_unk = tf.constant(value=0, dtype=tf.float32, shape=[2, emb_len])
        if init_mat is None:
            emb_mat_other = tf.get_variable('emb_mat',[dict_size - 2, emb_len], tf.float32)
        else:
            emb_mat_other = tf.get_variable("emb_mat",[dict_size - 2, emb_len], tf.float32,
                                            initializer=tf.constant_initializer(init_mat[2:], dtype=tf.float32,
                                                                                verify_shape=True))
        emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)

        if extra_mat is not None:
            if extra_trainable:
                extra_mat_var = tf.get_variable("extra_emb_mat",extra_mat.shape, tf.float32,
                                                initializer=tf.constant_initializer(extra_mat,
                                                                                    dtype=tf.float32,
                                                                                    verify_shape=True))
                return tf.concat([emb_mat, extra_mat_var], 0)
            else:
                #with tf.device('/cpu:0'):
                extra_mat_con = tf.constant(extra_mat, dtype=tf.float32)
                return tf.concat([emb_mat, extra_mat_con], 0)
        else:
            return emb_mat