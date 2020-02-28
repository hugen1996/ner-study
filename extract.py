# coding:UTF-8
import time
import torch
import numpy as np
import pickle as pkl
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='extract NER from text')
parser.add_argument('--model', type=str, default='bilstm_crf', help='choose a model: bilstm_crf or ....')
parser.add_argument('--text', type=str, required=True, help='input a text')
args = parser.parse_args()


def prepare_sequence(text, word_to_idx):
    return [word_to_idx.get(_, len(word_to_idx)-2) for _ in text]


if __name__ == '__main__':
    dataset = 'ResumeNER'  # 数据集
    x = import_module('models.' + args.model)
    config = x.Config(dataset, 'random')
    tag_to_idx = pkl.load(open(config.tag_path, 'rb'))
    word_to_idx = pkl.load(open(config.vocab_path, 'rb'))
    idx_to_tag = {_[1]: _[0] for _ in tag_to_idx.items()}
    config.tag_to_idx, config.word_to_idx = tag_to_idx, word_to_idx  # 在运行的时候进行赋值
    config.n_tag, config.n_vocab = len(tag_to_idx), len(word_to_idx)  # 运行的时候再进行赋值
    config.O_tag = 'O'
    config.O_tag_idx = config.tag_to_idx[config.O_tag]          # 运行的时候进行赋值

    text = args.text.strip()
    sentence = torch.tensor(prepare_sequence(text, word_to_idx)).unsqueeze(dim=0).to(config.device)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    with torch.no_grad():
        _, predict_tags = model(sentence)
        print(predict_tags[0])
        print([idx_to_tag[_] for _ in predict_tags[0]])
