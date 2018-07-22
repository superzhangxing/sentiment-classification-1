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
    def __init__(self, data_file_path, data_type, dicts = None):
        self.data_type = data_type
        _logger.add('building data set object for %s' % data_type)
        assert data_type in ['train', 'dev', 'test']
        # check
        if data_type in ['dev', 'test']:
            assert dicts is not None

        data_list = self.load_data(data_file_path, data_type)

        if data_type == 'train':
            self.dicts, self.max_lens = self.count_data_and_build_dict(data_list)
        else:
            _,self.max_lens = self.count_data_and_build_dict(data_list, False)
            self.dicts = dicts

        digitized_data_list = self.digitize_dataset(data_list, self.dicts, self.data_type)
        self.nn_data = digitized_data_list
        self.sample_num = len(self.nn_data)

        self.emb_mat_token, self.emb_mat_glove = None, None
        if data_type == 'train':
            self.emb_mat_token, self.emb_mat_glove = self.generate_index2vec_matrix()


    def save_dict(self, path):
        save_file(self.dicts, path,'token and char dict data', 'pickle')

    def load_data(self, data_file_path, data_type):
        _logger.add()
        _logger.add('load file for %s' % data_type)
        dataset = []
        with open(data_file_path, 'r', encoding = 'utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                dataset.append(json_obj)
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
            token_collection += sample['sentence']
            sent_len_collection += [len(sample['sentence'])]

            for token in sample['sentence']:
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
        else:
            dicts = {}
        _logger.done()
        return dicts, {'sent': max_sent_len, 'token': max_token_len}


    def digitize_dataset(self, data_list, dicts, data_type):
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
        _logger.add('digitizing data: %s...' % data_type)
        for sample in tqdm(data_list):
            sample['sentence_token_digitial'] = [digitize_token(token) for token in sample['sentence_token']]
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

