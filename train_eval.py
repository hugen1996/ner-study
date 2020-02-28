# coding: UTF-8
import torch
import torch.nn as nn
from sklearn import metrics
import time
import numpy as np
from utils import get_time_dif
from tensorboardX import SummaryWriter
from evaluate import evaluate


def init_network(model, method='xavier', exclude='embedding', seed=123):
    """
    训练参数权重初始化，默认为xavier
    :param model:
    :param method:
    :param exclude:
    :param seed:
    :return:
    """
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def get_train_acc(true_tags, predict_tags, seq_lens, remove_O_tag=False, O_tag_idx=None):
    flatten_tags = np.concatenate([_[:seq_lens[i]] for i, _ in enumerate(true_tags)])
    flatten_predict_tags = np.concatenate([_[:seq_lens[i]] for i, _ in enumerate(predict_tags)])
    if remove_O_tag:
        idx_list = (flatten_tags != O_tag_idx)
        flatten_tags = flatten_tags[idx_list]
        flatten_predict_tags = flatten_predict_tags[idx_list]
    return metrics.accuracy_score(flatten_tags, flatten_predict_tags)


def train(model, config, train_iter, dev_iter, test_iter):
    start_time = time.time()
    # 在训练中，必须采用model.train()模式
    # 采用Adam梯度下降方式
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    flag = False                                    # 用来决定是否提前停止训练
    total_batch = 0                                 # 记录进行到多少batch
    last_improve = 0                                # 记录上次验证集loss下降的batch数
    dev_best_loss = float('inf')                    # 测试集上的损失函数，初始值为inf

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (sentences, tags, seq_lens) in enumerate(train_iter):
            sentences, tags = sentences.to(config.device), tags.to(config.device)
            model.zero_grad()
            loss = model.neg_log_likelihood(sentences, tags)
            loss.backward()
            optimizer.step()
            # 每多少轮输出在训练集和验证集上的效果
            if total_batch % 10 == 0:
                _, predict_tags = model(sentences)
                true_tags = tags.data.cpu().numpy()
                train_acc = get_train_acc(true_tags, predict_tags, seq_lens, remove_O_tag=False)
                # train_acc = get_train_acc(true_tags, predict_tags, seq_lens,
                #                           remove_O_tag=True, O_tag_idx=config.O_tag_idx)
                dev_acc, dev_loss = evaluate(model, config, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, ' \
                      ' Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve >= config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(model, config, test_iter)


def test(model, config, test_iter):
    # 从模型中读取
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, config, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
