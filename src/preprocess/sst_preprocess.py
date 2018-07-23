from os.path import join as pjoin
from src.utils.file import save_file

class SSTPreprocess():
    def __init__(self, data_dir, label_class):
        self.label_class = label_class
        self.data_dir = data_dir
        self.trees, self.dictionary, self.sentiment_labels, self.dataset_split = self.load_sst_data(data_dir)
        self.samples = self.generate_samples()
        self.save_preprocess_file()

    # internal use
    def load_sst_data(self, data_dir):
        # dictionary
        dictionary = {}
        with open(pjoin(data_dir, 'dictionary.txt'), encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('|')
                assert len(line) == 2
                dictionary[line[0]] = int(line[1])

        # sentiment_labels
        sentiment_labels = {}
        with open(pjoin(data_dir, 'sentiment_labels.txt'), encoding='utf-8') as file:
            file.readline()  # for table head
            for line in file:
                line = line.strip().split('|')
                sent_float_value = float(line[1])

                sentiment_labels[int(line[0])] = sent_float_value

        # STree.txt and SOStr.txt
        trees = []
        with open(pjoin(data_dir, 'STree.txt'), encoding='utf-8') as file_STree, \
                open(pjoin(data_dir, 'SOStr.txt'), encoding='utf-8') as file_SOStr:
            for STree, SOStr in zip(file_STree, file_SOStr):
                sent_tree = []
                STree = list(map(int, STree.strip().split('|')))
                SOStr = SOStr.strip().split('|')

                for idx_t, parent_idx in enumerate(STree):
                    try:
                        token = SOStr[idx_t]
                        is_leaf = True
                        leaf_node_index_seq = [idx_t+1]
                    except IndexError:
                        token = ''
                        is_leaf = False
                        leaf_node_index_seq = []

                    new_node = {'node_index': idx_t+1, 'parent_index': parent_idx,
                                'token': token, 'is_leaf': is_leaf,
                                'leaf_node_index_seq': leaf_node_index_seq, }
                    sent_tree.append(new_node)

                # update leaf_node_index_seq
                idx_to_node_dict = dict((tree_node['node_index'], tree_node)
                                        for tree_node in sent_tree)
                for tree_node in sent_tree:
                    if not tree_node['is_leaf']: break
                    pre_node = tree_node
                    while pre_node['parent_index'] > 0:
                        cur_node = idx_to_node_dict[pre_node['parent_index']]
                        cur_node['leaf_node_index_seq'] += pre_node['leaf_node_index_seq']
                        cur_node['leaf_node_index_seq'] = list(
                            sorted(list(set(cur_node['leaf_node_index_seq']))))
                        pre_node = cur_node

                # update sentiment and add token_seq
                for tree_node in sent_tree:
                    tokens = [sent_tree[node_idx-1]['token'] for node_idx in tree_node['leaf_node_index_seq']]
                    phrase = ' '.join(tokens)
                    tree_node['sentiment_label'] = sentiment_labels[dictionary[phrase]]
                    tree_node['token_seq'] = tokens

                trees.append(sent_tree)

        # dataset_split (head)
        dataset_split = []
        with open(pjoin(data_dir, 'datasetSplit.txt'), encoding='utf-8') as file:
            file.readline()  # for table head
            for line in file:
                dataset_split.append(int(line.strip().split(',')[1]))

        return trees, dictionary, sentiment_labels, dataset_split

    def generate_samples(self):
        samples = []
        for _type, tree in zip(self.dataset_split, self.trees):
            for tree_node in tree:
                if tree_node['parent_index'] == 0:
                    sample = tree_node.copy()
                    break
            sample['data_type'] = _type
            sample['sentence_token'] = sample['token_seq'].copy()
            sample.pop('node_index')
            sample.pop('parent_index')
            sample.pop('token')
            sample.pop('leaf_node_index_seq')
            sample.pop('token_seq')
            sample.pop('is_leaf')
            samples.append(sample)

        # sentiment label fix to class
        fixed_samples = []
        if self.label_class == 2:
            for sample in samples:
                sentiment_label = sample['sentiment_label']
                if sentiment_label> 0.4 and sentiment_label <= 0.6:
                    continue
                if sentiment_label < 0.5:
                    sample['sentiment_int'] = 0
                else:
                    sample['sentiment_int'] = 1
                fixed_samples.append(sample)
        else:
            # 5 class
            for sample in samples:
                sentiment_label = sample['sentiment_label']
                if sentiment_label <= 0.2:
                    sentiment_int = 0
                elif sentiment_label <= 0.4:
                    sentiment_int = 1
                elif sentiment_label <= 0.6:
                    sentiment_int = 2
                elif sentiment_label <= 0.8:
                    sentiment_int = 3
                else:
                    sentiment_int = 4
                sample['sentiment_int'] = sentiment_int
                fixed_samples.append(sample)
        return fixed_samples

    def save_preprocess_file(self):
        train_samples = []
        dev_samples = []
        test_samples = []
        for sample in self.samples:
            if sample['data_type'] == 1:
                train_samples.append(sample)
            elif sample['data_type'] == 2:
                dev_samples.append(sample)
            else:
                test_samples.append(sample)

        save_file(train_samples, pjoin(self.data_dir, 'train.json'), mode = 'json')
        save_file(dev_samples, pjoin(self.data_dir, 'dev.json'), mode = 'json')
        save_file(test_samples, pjoin(self.data_dir, 'test.json'), mode = 'json')

sst_preprocess = SSTPreprocess(data_dir='/home/cike/zhangxing/Code/sentiment-classification-1/dataset/sst',
                            label_class=2)
samples = sst_preprocess.samples
# print('done')


