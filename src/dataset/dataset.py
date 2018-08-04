from configs import cfg
from src.utils.record_log import _logger
from src.utils.file import load_glove
from tqdm import tqdm
from src.utils.file import save_file
import random
import math
import json
import pickle
import numpy as np
from abc import abstractmethod


class Dataset(object):
    def __init__(self, train_file_path, dev_file_path = None, test_file_path=None):
        _logger.add('building data set object')

        train_data_list = self.load_data(train_file_path, 'train')
        dev_data_list = self.load_data(dev_file_path, 'dev')
        if test_file_path != None:
            test_data_list = self.load_data(test_file_path, 'test')

        data_list = []
        data_list.extend(train_data_list)
        data_list.extend(dev_data_list)
        if test_file_path != None:
            data_list.extend(test_data_list)

        self.dicts, self.max_lens = self.count_data_and_build_dict(data_list)

        self.digitized_train_data_list = self.digitize_dataset(train_data_list, self.dicts)
        self.digitized_dev_data_list = self.digitize_dataset(dev_data_list, self.dicts)
        if test_file_path != None:
            self.digitized_test_data_list = self.digitize_dataset(test_data_list, self.dicts)

        self.emb_mat_token, self.emb_mat_glove = self.generate_index2vec_matrix()


    def save_dict(self, path):
        save_file(self.dicts, path,'token and char dict data', 'pickle')

    def load_data(self, data_file_path, data_type):
        _logger.add()
        _logger.add('load file for %s' % data_type)
        dataset = []
        with open(data_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        _logger.done()
        return dataset

    def load_data_pickle(self, data_file_path, data_type):
        _logger.add()
        _logger.add('load file for %s' % data_type)
        dataset = None
        with open(data_file_path, 'rb', encoding = 'utf-8') as file:
            dataset = pickle.load(file)
        _logger.done()
        return dataset


    def count_data_and_build_dict(self, data_list, gene_dicts=True):
        def add_ept_and_unk(a_list):
            a_list.insert(0, '@@@empty')
            a_list.insert(1, '@@@unk')
            return a_list

        _logger.add()
        _logger.add('counting and build dictionaries')

        token_collection = []
        char_collection = []

        sent_len_collection = []
        token_len_collection = []

        for sample in data_list:
            token_collection += sample['sentence_token']
            sent_len_collection += [len(sample['sentence_token'])]

            for token in sample['sentence_token']:
                char_collection += list(token)
                token_len_collection.append(len(token))

        max_sent_len = max(sent_len_collection)
        max_token_len = max(token_len_collection)
        token_set = list(set(token_collection))
        char_set = list(set(char_collection))

        if gene_dicts:
            if cfg.use_glove_unk_token:
                glove_data = load_glove(cfg.word_embedding_length)
                glove_token_set = list(glove_data.keys())
                if cfg.lower_word:
                    token_set = list(token.lower() for token in token_set)
                    glove_token_set = list(set(token.lower() for token in glove_token_set))

                # delete token from glove_token_set which appears in token_set
                for token in token_set:
                    try:
                        glove_token_set.remove(token)
                    except ValueError:
                        pass
            else:
                if cfg.lower_word:
                    token_set = list(token.lower() for token in token_set)
                glove_token_set = []
            token_set = add_ept_and_unk(token_set)
            char_set = add_ept_and_unk(char_set)
            dicts = {'token': token_set, 'char': char_set, 'glove': glove_token_set}

        return dicts, {'sent': max_sent_len, 'token': max_token_len}


    def digitize_dataset(self, data_list, dicts):
        token2index = dict([(token, idx) for idx, token in enumerate(dicts['token']+dicts['glove'])])
        char2index = dict([(char, idx) for idx, char in enumerate(dicts['char'])])

        def digitize_token(token):
            token = token if not cfg.lower_word else token.lower()
            try:
                return token2index[token]
            except KeyError:
                return 1

        def digitize_char(char):
            try:
                return char2index[char]
            except KeyError:
                return 1

        _logger.add()
        _logger.add('digitizing data' )
        for sample in tqdm(data_list):
            sample['sentence_token_digital'] = [digitize_token(token) for token in sample['sentence_token']]
            sample['sentence_char_digital'] = [[digitize_char(char) for char in list(token)]
                                               for token in sample['sentence_token']]

        _logger.done()
        return data_list

    def generate_index2vec_matrix(self):
        _logger.add()
        _logger.add('generate index to vector numpy matrix')

        token2vec = load_glove(cfg.word_embedding_length)
        if cfg.lower_word:
            newtoken2vec = {}
            for token, vec in token2vec.items():
                newtoken2vec[token.lower()] = vec
            token2vec = newtoken2vec

        # prepare data from trainDataset and devDataset
        mat_token = np.random.uniform(-0.05, 0.05, size=(len(self.dicts['token']), cfg.word_embedding_length)).astype(
            cfg.floatX)

        mat_glove = np.zeros((len(self.dicts['glove']), cfg.word_embedding_length), dtype = cfg.floatX)

        for idx, token in enumerate(self.dicts['token']):
            try:
                mat_token[idx] = token2vec[token]
            except KeyError:
                pass
            mat_token[0] = np.zeros(shape=(cfg.word_embedding_length,), dtype=cfg.floatX)

        for idx, token in enumerate(self.dicts['glove']):
            mat_glove[idx] = token2vec[token]

        _logger.add('Done')
        return mat_token, mat_glove

    @staticmethod
    def generate_batch_sample_iter(digitalized_data = None, max_step=None):
        if max_step is not None:
            batch_size = cfg.train_batch_size

            def data_queue(data, batch_size):
                assert len(data) >= batch_size
                random.shuffle(data)
                data_ptr = 0
                dataRound = 0
                idx_b = 0
                step = 0
                while True:
                    if data_ptr + batch_size <= len(data):
                        yield data[data_ptr:data_ptr + batch_size], dataRound, idx_b
                        data_ptr += batch_size
                        idx_b += 1
                        step += 1
                    elif data_ptr + batch_size > len(data):
                        offset = data_ptr + batch_size - len(data)
                        out = data[data_ptr:]
                        random.shuffle(data)
                        out += data[:offset]
                        data_ptr = offset
                        dataRound += 1
                        yield out, dataRound, 0
                        idx_b = 1
                        step += 1
                    if step >= max_step:
                        break
            batch_num = math.ceil(len(digitalized_data) / batch_size)
            for sample_batch, data_round, idx_b in data_queue(digitalized_data, batch_size):
                yield sample_batch, batch_num, data_round, idx_b
        else:
            batch_size = cfg.test_batch_size
            batch_num = math.ceil(len(digitalized_data) / batch_size)
            idx_b = 0
            sample_batch = []
            for sample in digitalized_data:
                sample_batch.append(sample)
                if len(sample_batch) == batch_size:
                    yield sample_batch, batch_num, 0, idx_b
                    idx_b += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield sample_batch, batch_num, 0, idx_b
