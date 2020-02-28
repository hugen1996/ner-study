import os
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl
import time
import math
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
from models.bilstm_crf import Config

MAX_VOCAB_SIZE = 10000  # 限制词表长度
O_TAG = 'O'
UNK, PAD = '<UNK>', '<PAD>'     # 未知字和填充
START_TAG, STOP_TAG = '<START>', '<STOP>'  # 开始和结束标志


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))


def build_dic(file_path, max_size, min_freq=None):
    """
    从训练集中构建vocab和tag字典
    voacb和tag按频率排列
    :param file_path:
    :param max_size:
    :return:
    """
    tag_dic = {}
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            word, tag = line.split()
            tag_dic[tag] = tag_dic.get(tag, 0) + 1
            vocab_dic[word] = vocab_dic.get(word, 0)+1

    tag_list = sorted([_ for _ in tag_dic.items()], key=lambda x: x[1], reverse=True)
    vocab_list = sorted([_ for _ in vocab_dic.items()], key=lambda x: x[1], reverse=True)[:max_size]

    tag_dic = {item[0]: idx for idx, item in enumerate(tag_list)}
    tag_dic.update({START_TAG: len(tag_dic), STOP_TAG: len(tag_dic) + 1})
    vocab_dic = {item[0]: idx for idx, item in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic)+1})

    return vocab_dic, tag_dic


class MyDataSet(Dataset):
    """
    训练中所用到的数据集
    """
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.tags[index]


def build_dataset(config):
    """
    函数功能： 构建训练、验证和测试用的dataset
    :param config:
    :return:
    """

    """step:1"""
    """"首先需要获得word_to_idx和tag_to_idx"""
    if os.path.exists(config.vocab_path):
        tag_to_idx = pkl.load(open(config.tag_path, 'rb'))
        word_to_idx = pkl.load(open(config.vocab_path, 'rb'))
    else:
        word_to_idx, tag_to_idx = build_dic(config.train_path, max_size=MAX_VOCAB_SIZE)
        pkl.dump(tag_to_idx, open(config.tag_path, 'wb'))
        pkl.dump(word_to_idx, open(config.vocab_path, 'wb'))
    # print(len(word_to_idx), len(tag_to_idx))
    # print(word_to_idx, tag_to_idx)
    config.n_tag, config.n_vocab = len(tag_to_idx), len(word_to_idx)
    config.tag_to_idx, config.word_to_idx = tag_to_idx, word_to_idx
    print(f"vocab size: {config.n_vocab}, tag size: {config.n_tag}")
    config.O_tag_idx = tag_to_idx[config.O_tag]

    """step:2"""
    """然后开始构建dataset"""
    def load_dataset(path):
        tags_list = []
        sentences_list = []
        with open(path, 'r', encoding='UTF-8') as f:
            tags = []
            sentence = []
            for line in f:
                line = line.strip()
                if not line:
                    """
                    我们在每个tags和sentence的最后都加上一个O_TAG和PAD，
                    这是为了方便collate_fn函数中进行pad操作
                    """
                    tags.append(tag_to_idx[O_TAG])
                    sentence.append(word_to_idx[PAD])
                    tags_list.append(tags)
                    sentences_list.append(sentence)
                    tags, sentence = [], []
                    continue
                word, tag = line.split()
                tags.append(tag_to_idx[tag])
                sentence.append(word_to_idx.get(word, word_to_idx[UNK]))
        return MyDataSet(sentences_list, tags_list)

    """返回train_dataset, dev_dataset, test_dataset"""
    train_dataset = load_dataset(config.train_path)
    dev_dataset = load_dataset(config.dev_path)
    test_dataset = load_dataset(config.test_path)

    # 根据train_dataset的大小来确定config中的require_improvement
    # 在这里设定为：如果在训练集在上遍历一次后accuracy没有提高，那么停止训练
    config.require_improvement = math.ceil(len(train_dataset)/config.batch_size)
    # config.require_improvement = 2

    return train_dataset, dev_dataset, test_dataset, word_to_idx, tag_to_idx


def build_dataiter(dataset, batch_size, shuffle, num_workers=0):
    def collate_func(batch):
        """
        对每一批次的batch进行处理
        注意每个sentence的最后都加上了一个PAD，每个tags的最后都加上了一个O_TAG
        处理的时候需要去掉
        :param batch:
        :param data:
        :return:
        """
        batch.sort(key=lambda _: len(_[0]), reverse=True)
        batch_length = [len(_[0]) - 1 for _ in batch]
        sentences_list, tags_list = [], []
        for i, (sentence, tags) in enumerate(batch):
            tags.extend([tags[-1]] * (batch_length[0] - batch_length[i]))
            sentence.extend([sentence[-1]] * (batch_length[0] - batch_length[i]))

            tags_list.append(tags[:-1])
            sentences_list.append(sentence[:-1])

        return torch.tensor(sentences_list, dtype=torch.long), torch.tensor(tags_list, dtype=torch.long), batch_length

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_func, num_workers=num_workers)


if __name__ == '__main__':
    # 构建voacb_dic和tag_dic
    tag_path = "./ResumeNER/pkl/tag.pkl"
    vocab_path = "./ResumeNER/pkl/vocab.pkl"
    train_path = "./ResumeNER/data/train.char.bmes"
    if os.path.exists(vocab_path):
        tag_to_idx = pkl.load(open(tag_path, 'rb'))
        word_to_idx = pkl.load(open(vocab_path, 'rb'))
    else:
        word_to_idx, tag_to_idx = build_dic(train_path, max_size=MAX_VOCAB_SIZE)
        pkl.dump(tag_to_idx, open(tag_path, 'wb'))
        pkl.dump(word_to_idx, open(vocab_path, 'wb'))

    # print(len(word_to_idx), len(tag_to_idx))
    # print(word_to_idx, tag_to_idx)

    # 构建预训练的词向量
    embed_dim = 300
    pretrain_path = "./ResumeNER/data/sgns.sogou.char"
    embeddings_trimmed_path = "./ResumeNER/npz/sogou_embeddings"
    embeddings = np.random.randn(len(word_to_idx), embed_dim)
    with open(pretrain_path, 'r', encoding='UTF-8') as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:  # 如果第一行是标题，那么跳过，视情况而定
                continue
            lin = line.strip().split(" ")
            if lin[0] in word_to_idx:
                idx = word_to_idx[lin[0]]
            embed = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(embed, dtype='float64')
    np.savez_compressed(embeddings_trimmed_path, embeddings=embeddings)
    config = Config('ResumeNER', 'random')
    # train_dataset, dev_dataset, test_dataset, word_to_idx, tag_to_idx = build_dataset(config)
    # print(train_dataset[0])
    # loader = build_dataiter(train_dataset, batch_size=8, shuffle=False)
    # for _, iter in enumerate(loader):
    #     print(iter[0], iter[1])
    #     break
