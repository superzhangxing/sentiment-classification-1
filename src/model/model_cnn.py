from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf

from src.model.model_template import ModelTemplate
from src.nn_utils.integration_func import generate_embedding_mat
from src.nn_utils.nn import linear
from src.nn_utils.cnn import cnn_Kim

class ModelCNN(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        super(ModelCNN, self).__init__(token_emb_mat, glove_emb_mat, tds, cds, tl, scope)
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        _logger.add()
        _logger.add('building %s neural network structure ...' % cfg.network_type)
        tds, cds = self.tds, self.cds
        tl = self.tl
        tel, cel, cos, ocd, fh = self.tel, self.cel, self.cos, self.ocd, self.fh
        hn = self.hn
        bs, sl = self.bs, self.sl
        self.l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            # self.W = tf.Variable(
            #     tf.random_uniform([self.token_emb_mat.shape[0], self.token_emb_mat.shape[1]], -1.0, 1.0),
            #     name = 'W'
            # )
            # self.W = tf.Variable(
            #     initial_value=self.token_emb_mat,
            #     name = 'W'
            # )
            self.W = tf.constant(self.token_emb_mat)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.token_seq)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.device('/gpu:0'):
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, (filter_size, out_channel) in enumerate(zip(self.fh, self.ocd)):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.tel, 1, out_channel]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[out_channel]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") # bs, sl-filter_size+1, 1, out_channel
                    h = tf.squeeze(h, 2) # bs, sl-filter_size+1, oc
                    # Maxpooling over the outputs
                    pooled = tf.reduce_max(h, 1)
                    pooled_outputs.append(pooled)

            self.h_pool = tf.concat(pooled_outputs, 1)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool, cfg.dropout)

            # Final (unnormalized) scores and predictions
            num_filters_total = tf.reduce_sum(self.ocd)
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[300, self.output_class],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[self.output_class]), name="b")
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
                self.predictions = tf.argmax(self.logits, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.gold_label)
                self.loss = tf.reduce_mean(losses) + cfg.wd * self.l2_loss
                tf.summary.scalar('loss', self.loss)

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.gold_label, tf.cast(tf.argmax(self.logits, -1), tf.int32))
                self.accuracy = tf.cast(correct_predictions, "float")

            # optimizer
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # with tf.variable_scope('emb'):
        #     token_emb_mat = generate_embedding_mat(tds, tel, init_mat=self.token_emb_mat,
        #                                            extra_mat=self.glove_emb_mat, extra_trainable=self.finetune_emb,
        #                                            scope='gene_token_emb_mat')
        #     emb = tf.nn.embedding_lookup(token_emb_mat, self.token_seq) # bs, sl, tel
        #     self.tensor_dict['emb'] = emb
        #
        # with tf.variable_scope('cnn'):
        #     sent_rep = cnn_Kim(emb, filter_height = self.fh, out_channel_dims=self.ocd, scope='cnn_Kim', keep_prob=cfg.dropout,
        #                        is_train=self.is_train,activation='relu', tensor_dict=self.tensor_dict, name='')
        #     self.tensor_dict['sent_rep'] = sent_rep
        #
        # with tf.variable_scope('output'):
        #     logits = tf.nn.relu(linear([sent_rep], self.output_class, True, scope='pre_logits_linear',
        #                                   input_keep_prob=cfg.dropout,
        #                                   is_train=self.is_train))  # bs, hn
        #     # sent_rep = tf.cond(self.is_train, lambda: tf.nn.dropout(sent_rep, cfg.dropout), lambda: sent_rep)
        #     # logits = tf.layers.dense(sent_rep, self.output_class, tf.nn.relu)
        # return logits


