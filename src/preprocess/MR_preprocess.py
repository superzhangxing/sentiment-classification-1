import os
import re
import random
from src.utils.file import save_file
from os.path import join as pjoin

class MRPreprocess(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_pos = os.path.join(data_dir, 'rt-polarity.pos-utf8')
        self.file_neg = os.path.join(data_dir, 'rt-polarity.neg-utf8')
        self.samples_pos, self.samples_neg = self.preprocess()
        self.fixed_samples = self.generate_samples()
        self.save_preprocess_file()


    def preprocess(self):
        space_token = re.compile(r' +')
        samples_pos = []
        samples_neg = []
        with open(self.file_pos, 'rt') as fd:
            for line_id, line in enumerate(fd):
                line = line.replace('é', 'e')
                line = line.replace('è', 'e')
                line = line.replace('ï', 'i')
                line = line.replace('í', 'i')
                line = line.replace('ó', 'o')
                line = line.replace('ô', 'o')
                line = line.replace('ö', 'o')
                line = line.replace('á', 'a')
                line = line.replace('â', 'a')
                line = line.replace('ã', 'a')
                line = line.replace('à', 'a')
                line = line.replace('ü', 'u')
                line = line.replace('û', 'u')
                line = line.replace('ñ', 'n')
                line = line.replace('ç', 'c')
                line = line.replace('æ', 'ae')
                line = line.replace('\xa0', ' ')
                line = line.replace('\xc2', '')

                line = line.replace('‘', '\'')
                line = line.replace('’', '\'')
                line = line.replace('\'', ' \' ')
                line = line.replace('\' s ', ' \'s ')
                line = line.replace('n \' t ', ' n\'t ')
                line = line.replace('\' ve ', ' \'ve ')
                line = line.replace('\' ll ', ' \'ll ')
                line = line.replace('\' re ', ' \'re ')

                line = line.replace('[', ' [ ')
                line = line.replace(']', ' ] ')

                # 去掉多余空格
                line = space_token.sub(' ', line)
                # 去掉末尾的空格,\n
                if (len(line) > 2 and line[-2] == ' '):
                    line = line[:-2]
                else:
                    line = line[:-1]

                samples_pos.append(line)

        with open(self.file_neg, 'rt') as fd:
            for line_id, line in enumerate(fd):
                line = line.replace('é', 'e')
                line = line.replace('è', 'e')
                line = line.replace('ï', 'i')
                line = line.replace('í', 'i')
                line = line.replace('ó', 'o')
                line = line.replace('ô', 'o')
                line = line.replace('ö', 'o')
                line = line.replace('á', 'a')
                line = line.replace('â', 'a')
                line = line.replace('ã', 'a')
                line = line.replace('à', 'a')
                line = line.replace('ü', 'u')
                line = line.replace('û', 'u')
                line = line.replace('ñ', 'n')
                line = line.replace('ç', 'c')
                line = line.replace('æ', 'ae')
                line = line.replace('\xa0', ' ')
                line = line.replace('\xc2', '')

                line = line.replace('‘', '\'')
                line = line.replace('’', '\'')
                line = line.replace('\'', ' \' ')
                line = line.replace('\' s ', ' \'s ')
                line = line.replace('n \' t ', ' n\'t ')
                line = line.replace('\' ve ', ' \'ve ')
                line = line.replace('\' ll ', ' \'ll ')
                line = line.replace('\' re ', ' \'re ')

                line = line.replace('[', ' [ ')
                line = line.replace(']', ' ] ')

                # 去掉多余空格
                line = space_token.sub(' ', line)
                # 去掉末尾的空格,\n
                if (len(line) > 2 and line[-2] == ' '):
                    line = line[:-2]
                else:
                    line = line[:-1]

                samples_neg.append(line)
        return samples_pos, samples_neg

    def generate_samples(self):
        fixed_samples = []
        for sample in self.samples_pos:
            fixed_sample = {}
            fixed_sample['sentence_token'] = sample.split()
            fixed_sample['sentiment_int'] = 1
            fixed_samples.append(fixed_sample)

        for sample in self.samples_neg:
            fixed_sample = {}
            fixed_sample['sentence_token'] = sample.split()
            fixed_sample['sentiment_int'] = 0
            fixed_samples.append(fixed_sample)

        return fixed_samples

    def save_preprocess_file(self):
        random.shuffle(self.fixed_samples)
        dev_rate = 0.1
        train_len = int(len(self.fixed_samples)*(1-dev_rate))
        train = self.fixed_samples[:train_len]
        dev = self.fixed_samples[train_len:]
        save_file(train, pjoin(self.data_dir, 'train.json'), mode = 'json')
        save_file(dev, pjoin(self.data_dir, 'dev.json'), mode = 'json')

sst_preprocess = MRPreprocess(data_dir='/home/cike/zhangxing/Code/sentiment-classification-1/dataset/MR')