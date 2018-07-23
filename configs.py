import argparse
import os
from os.path import join
from src.utils.time_counter import TimeCounter

class Configs(object):
    def __init__(self):
        self.project_dir = os.getcwd()
        self.dataset_dir = join(self.project_dir, 'dataset')

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))

        # @ ----- control ----
        parser.add_argument('--debug', type='bool', default=False, help='whether run as debug mode')
        parser.add_argument('--mode', type=str, default='train', help='train, dev or test')
        parser.add_argument('--network_type', type=str, default='test', help='network type')
        parser.add_argument('--log_period', type=int, default=100, help='save tf summary period')  ###  change for running
        parser.add_argument('--save_period', type=int, default=100, help='abandoned')
        parser.add_argument('--eval_period', type=int, default=100, help='evaluation period')  ###  change for running
        parser.add_argument('--gpu', type=int, default=3, help='employed gpu index')
        parser.add_argument('--gpu_mem', type=float, default=0.96, help='gpu memory ratio to employ')
        parser.add_argument('--model_dir_suffix', type=str, default='', help='model folder name suffix')
        parser.add_argument('--load_path', type=str, default=None, help='specify which pre-trianed model to be load')
        parser.add_argument('--load_model', type=str, default=False, help='load pretrained model')

        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=200, help='max epoch number')
        parser.add_argument('--train_batch_size', type=int, default=50, help='Train Batch Size')
        parser.add_argument('--test_batch_size', type=int, default=100, help='Test Batch Size')
        parser.add_argument('--optimizer', type=str, default='adam', help='choose an optimizer[adadelta|adam]')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Init Learning rate')


        # @ ----- Text Processing ----
        parser.add_argument('--word_embedding_length', type=int, default=300, help='word embedding length')
        parser.add_argument('--char_embedding_length', type=int, default=300, help='char embedding length')
        parser.add_argument('--char_out_size', type=int, default=20, help='char out size')
        parser.add_argument('--glove_corpus', type=str, default='6B', help='choose glove corpus to employ')
        parser.add_argument('--use_glove_unk_token', type='bool', default=True, help='')
        parser.add_argument('--lower_word', type='bool', default=True, help='')
        parser.add_argument('--out_channel_dims', type=str, default='100,100,100', help='out channel dims')
        parser.add_argument('--filter_heights', type=str, default='3,4,5', help='filter heights')
        parser.add_argument('--fine_tune', type='bool', default=False, help='fine tune extra embedding mat')


        # @ ------neural network-----
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep prob')
        parser.add_argument('--hidden_units_num', type=int, default=300, help='Hidden units number of Neural Network')

        # @ ------task-----
        parser.add_argument('--output_class', type=int, default=2, help='output class')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))

        # ------- name --------
        self.train_data_name = 'train.json'
        self.dev_data_name = 'dev.json'
        self.test_data_name = 'test.json'
        self.infer_train_data_name = 'train'
        self.infer_dev_data_name = 'dev'
        self.infer_test_data_name = 'test'

        self.processed_name = 'processed' + self.get_params_str(['lower_word', 'use_glove_unk_token',
                                                                 'glove_corpus', 'word_embedding_length']) + '.pickle'
        self.dict_name = 'dicts' + self.get_params_str(['lower_word', 'use_glove_unk_token',
                                                        ])

        self.model_name = self.network_type
        self.model_ckpt_name = 'modelfile.ckpt'



        # ---------- dir -------------
        self.data_dir = join(self.dataset_dir, 'sst')
        self.glove_dir = join(self.dataset_dir, 'glove')
        self.result_dir = self.mkdir(self.project_dir, 'result')
        self.standby_log_dir = self.mkdir(self.result_dir, 'log')
        self.dict_dir = self.mkdir(self.result_dir, 'dict')
        self.processed_dir = self.mkdir(self.result_dir, 'processed_data')
        self.infer_dir = self.mkdir(self.result_dir, 'infer')

        self.log_dir = None
        self.all_model_dir = self.mkdir(self.result_dir, 'model')
        self.model_dir = self.mkdir(self.all_model_dir, self.model_name)
        self.log_dir = self.mkdir(self.model_dir, 'log_files')
        self.summary_dir = self.mkdir(self.model_dir, 'summary')
        self.ckpt_dir = self.mkdir(self.model_dir, 'ckpt')

        # -------- path --------
        self.train_data_path = join(self.data_dir, self.train_data_name)
        self.dev_data_path = join(self.data_dir, self.dev_data_name)
        self.test_data_path = join(self.data_dir, self.test_data_name)
        self.infer_train_data_path = join(self.infer_dir, self.infer_train_data_name)
        self.infer_dev_data_path = join(self.infer_dir, self.infer_dev_data_name)
        self.infer_test_data_path = join(self.infer_dir, self.infer_test_data_name)

        self.processed_path = join(self.processed_dir, self.processed_name)
        self.dict_path = join(self.dict_dir, self.dict_name)
        self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)

        self.extre_dict_path = join(self.dict_dir, 'extra_dict.json')

        # dtype
        self.floatX = 'float32'
        self.intX = 'int32'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        self.time_counter = TimeCounter()

    def get_params_str(self, params):
        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '_' + str(eval('self.args.' + paramsStr))
        return model_params_str

    def mkdir(self, *args):
        dirPath = join(*args)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        return dirPath

    def get_file_name_from_path(self, path):
        assert isinstance(path, str)
        fileName = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return fileName


cfg = Configs()
