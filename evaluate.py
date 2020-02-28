# coding: UTF-8
import numpy as np
import torch
from sklearn import metrics


def evaluate(model, config, data_iter, remove_O_tag=False, test=False):
    """
    对命名实体识别的结果进行评估
    :param config:
    :param model:
    :param data_iter:
    :param remove_O_tag:
    :param test:
    :return:
    """
    # 所有的评估必须采用model.eval()模式
    model.eval()
    loss_total = 0.0
    tags_all = np.array([], dtype=int)
    predict_tags_all = np.array([], dtype=int)
    with torch.no_grad():
        for sentences, tags, seq_lens in data_iter:
            sentences, tags = sentences.to(config.device), tags.to(config.device)
            # neg_log_likelihood为损失函数
            loss = model.neg_log_likelihood(sentences, tags)
            loss_total += loss
            # forward函数返回score和tags
            _, predict_tags = model(sentences)

            true_tags = tags.data.cpu().numpy()
            flatten_tags = np.concatenate([_[:seq_lens[i]] for i, _ in enumerate(true_tags)])
            flatten_predict_tags = np.concatenate([_[:seq_lens[i]] for i, _ in enumerate(predict_tags)])

            tags_all = np.concatenate([tags_all, flatten_tags])
            predict_tags_all = np.concatenate([predict_tags_all, flatten_predict_tags])

    if remove_O_tag:
        idx_list = (tags_all != config.O_tag_idx)
        tags_all = tags_all[idx_list]
        predict_tags_all = predict_tags_all[idx_list]
    acc = metrics.accuracy_score(tags_all, predict_tags_all)

    if test:
        # target_names = [_[0] for _ in config.tag_to_idx.items()]
        # if remove_O_tag:
        #     target_names.remove(config.O_tag)
        report = metrics.classification_report(tags_all, predict_tags_all, digits=4)
        confusion = metrics.confusion_matrix(tags_all, predict_tags_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
