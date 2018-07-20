from configs import cfg
from src.utils.record_log import _logger
import numpy as np
import pickle
import os
import json

def save_file(data, file_path, data_name = 'data', mode = 'pickle'):
    _logger.add()
    _logger.add('saving %s to %s' % (data_name, file_path))

    if mode == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(obj = data, file = f)
    elif mode == 'json':
        with open(file_path, 'w', encoding = 'utf-8') as f:
            json.dump(obj = data, fp = f)
    else:
        raise('ValueError', 'Function save_file does not have mode %s' % (mode))

    _logger.add('Done')

def load_file(file_path, data_name = 'data', mode = 'pickle'):
    _logger.add()
    _logger.add('Trying to load %s from %s' % (data_name, file_path))
    data = None
    if_load = False

    if os.path.isfile(file_path):
        _logger.add('Have found the file, loading... ')

        if mode == 'pickle':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if_load = True
        elif mode == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if_load = True
        else:
            raise(ValueError, 'Function save_file does not have mode %s' % (mode))

    else:
        _logger.add('Have not found the file')
    _logger.add('Done')

    return (if_load, data)

def load_glove(dim):
    _logger.add()
    _logger.add('loading glove from pre-trained file...')
    # if dim not in [50, 100, 200, 300]:
    #     raise(ValueError, 'glove dim must be in [50, 100, 200, 300]')
    word2vec = {}
    with open(os.path.join(cfg.glove_dir, "glove.%s.%sd.txt" % (cfg.glove_corpus, str(dim))), encoding='utf-8') as f:
        for line in f:
            line_lst = line.split()
            word = "".join(line_lst[0:-dim])
            vector = list(map(float, line_lst[-dim:]))
            word2vec[word] = vector

    _logger.add('Done')
    return word2vec
