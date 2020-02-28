# coding:UTF-8
import time
import torch
import numpy as np
from tqdm import tqdm
from train_eval import train, init_network
from importlib import import_module
from utils import build_dataset, build_dataiter, get_time_dif
import argparse

parser = argparse.ArgumentParser(description='Chinese Named Entity Recognition')
parser.add_argument('--model', type=str, required=True, help='choose a model: bilstm_crf or ....')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'ResumeNER'  # 数据集

    # 搜狗新闻: embedding_SougouNews.npz, 腾讯: embedding_Tencent.npz, 随机初始化: random
    embedding = 'sogou_embeddings.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    # 为了保证每次的结果相同
    # 在下面设定一些参数
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    # 构建数据集
    start_time = time.time()
    print('Loading data...')
    train_dataset, dev_dataset, test_dataset, word_to_idx, tag_to_idx = build_dataset(config)
    print(F"train dataset size: {len(train_dataset)}, dev dataset size: {len(dev_dataset)}, test dataset size:{len(test_dataset)}")
    train_iter = build_dataiter(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_iter = build_dataiter(dev_dataset, batch_size=config.batch_size, shuffle=True)
    test_iter = build_dataiter(test_dataset, batch_size=config.batch_size, shuffle=False)
    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

    # 进行训练
    model = x.Model(config).to(config.device)
    print(model.parameters)
    train(model=model, config=config, train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter)
