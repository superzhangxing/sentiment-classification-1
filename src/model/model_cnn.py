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

        with tf.variable_scope('emb'):
            token_emb_mat = generate_embedding_mat(tds, tel, init_mat=self.token_emb_mat,
                                                   extra_mat=self.glove_emb_mat, extra_trainable=self.finetune_emb,
                                                   scope='gene_token_emb_mat')
            emb = tf.nn.embedding_lookup(token_emb_mat, self.token_seq) # bs, sl, tel
            self.tensor_dict['emb'] = emb

        with tf.variable_scope('cnn'):
            sent_rep = cnn_Kim(self.token_seq, filter_size=3, scope='cnn_Kim', keep_prob=cfg.dropout,
                               is_train=self.is_train,activation='relu', tensor_dict=self.tensor_dict, name='')
            self.tensor_dict['sent_rep'] = sent_rep

        with tf.variable_scope('output'):
            pre_output = tf.nn.relu(linear([sent_rep], hn, True, 0., scope='pre_output', squeeze=False,
                                           input_keep_prob=cfg.dropout, is_train=self.is_train))
            logits = linear([pre_output], self.output_class, True, 0., scope='logits', squeeze=False,
                            input_keep_prob=cfg.dropout, is_train=self.is_train)
            self.tensor_dict['logits'] = logits

        return logits


