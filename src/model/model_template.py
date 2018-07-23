from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        self.scope = scope
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.token_emb_mat, self.glove_emb_mat = token_emb_mat, glove_emb_mat

        # ---- place holder ----
        self.token_seq = tf.placeholder(tf.int32, [None, None], name='token_seq')
        self.char_seq = tf.placeholder(tf.int32, [None, None, tl], name='context_char')

        self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label') # bs
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')


        # ---------- parameters ----------
        self.tds, self.cds = tds, cds
        self.tl = tl
        self.tel = cfg.word_embedding_length
        self.cel = cfg.char_embedding_length
        self.cos = cfg.char_out_size
        self.ocd = list(map(int, cfg.out_channel_dims.split(',')))
        self.fh = list(map(int, cfg.filter_heights.split(',')))
        self.hn = cfg.hidden_units_num
        self.finetune_emb = cfg.fine_tune

        self.output_class = cfg.output_class

        self.bs = tf.shape(self.token_seq)[0]
        self.sl = tf.shape(self.token_seq)[1]

        # ---------- other -----------
        self.token_mask = tf.cast(self.token_seq, tf.bool)
        self.char_mask = tf.cast(self.char_seq, tf.bool)
        self.token_len = tf.reduce_sum(tf.cast(self.token_mask, tf.int32), -1)
        self.char_len = tf.reduce_sum(tf.cast(self.char_mask, tf.int32), -1)

        self.tensor_dict = {}

        # ---------- start ----------
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.summary = None
        self.opt = None
        self.train_op = None

    @abstractmethod
    def build_network(self):
        pass

    def build_loss(self):
        # trainable variables number
        # weight_decay
        with tf.name_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), cfg.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)
        _logger.add('regularization var num: %d' % len(reg_vars))
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        _logger.add('trainable var num: %d' % len(trainable_vars))

        # compute loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.gold_label,
            logits=self.logits
        )
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.gold_label
        ) # [bs]
        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):
        self.logits = self.build_network()
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()

        # ---------- ema ----------

        self.summary = tf.summary.merge_all()
        # ---------- optimization ----------
        if cfg.optimizer.lower() == 'adadelta':
            # assert cfg.learning_rate > 0.1 and cfg.learning_rate < 1.
            self.opt = tf.train.AdadeltaOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            # assert cfg.learning_rate < 0.1
            self.opt = tf.train.AdamOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            # assert cfg.learning_rate < 0.1
            self.opt = tf.train.RMSPropOptimizer(cfg.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        all_params_num = 0
        for elem in trainable_vars:
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'): continue
            params_num = 1
            for l in elem.get_shape().as_list(): params_num *= l
            all_params_num += params_num
        _logger.add('Trainable Parameters Number: %d' % all_params_num)

        self.train_op = self.opt.minimize(self.loss, self.global_step,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))

    def get_feed_dict(self, sample_batch, data_type='train'):
        # max sentence len
        sl = 0

        for sample in sample_batch:
            sl = max(sl, len(sample['sentence_token_digital']))

        # token and char
        token_seq_b = []
        char_seq_b = []

        for sample in sample_batch:
            token_seq = np.zeros([sl], cfg.intX)
            char_seq = np.zeros([sl, self.tl], cfg.intX)
            for idx_t, (token, char_seq_v) in enumerate(zip(sample['sentence_token_digital'],
                                                            sample['sentence_char_digital'])):
                token_seq[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.tl:
                        char_seq[idx_t, idx_c] = char
            token_seq_b.append(token_seq)
            char_seq_b.append(char_seq)
        token_seq_b = np.stack(token_seq_b)
        char_seq_b = np.stack(char_seq_b)


        # label
        gold_label_b = []
        for sample in sample_batch:
            gold_label_int = sample['sentiment_int']
            gold_label_b.append(gold_label_int)
        gold_label_b = np.stack(gold_label_b).astype(cfg.intX)

        feed_dict = {
            self.token_seq: token_seq_b, self.char_seq:char_seq_b,
            self.gold_label: gold_label_b,
            self.is_train: True if data_type == 'train' else False
        }
        return feed_dict

    def step(self, sess, batch_samples, get_summary=False):
        # assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'train')
        cfg.time_counter.add_start()
        if get_summary:
            loss, summary, train_op = sess.run([self.loss, self.summary, self.train_op],
                                               feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        cfg.time_counter.add_stop()
        return loss, summary, train_op
