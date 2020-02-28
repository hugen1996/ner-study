from abc import ABC
import numpy as np
import torch
import torch.nn as nn

START_TAG, STOP_TAG = '<START>', '<STOP>'  # 开始和结束标志


class Config(object):
    """
    有关模型的配置
    """
    def __init__(self, dataset, embedding='random'):
        self.model_name = "BiLSTM_CRF"                                                      # 模型名
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        # cpu or cuda
        # 有关训练集、验证集、测试集的路径
        self.train_path = dataset + '/data/train.char.bmes'                                 # 训练集
        self.dev_path = dataset + '/data/dev.char.bmes'                                     # 验证集
        self.test_path = dataset + '/data/test.char.bmes'                                   # 测试集

        # tag和vocab的路径设置
        self.tag_path = dataset + '/pkl/tag.pkl'
        self.vocab_path = dataset + '/pkl/vocab.pkl'
        self.tag_to_idx = None         # 在运行的时候进行赋值
        self.word_to_idx = None        # 在运行的时候进行赋值
        self.n_tag = 0  # 运行的时候再进行赋值
        self.n_vocab = 0  # 运行的时候再进行赋值
        self.O_tag = 'O'
        self.O_tag_idx = None   # 运行的时候进行赋值

        # 日志和模型的存储
        self.log_path = dataset + '/log/' + self.model_name
        self.save_path = dataset + '/saved_model/' + self.model_name + '.ckpt'

        # embedding层的设置
        self.embedding_pretrained = torch.tensor(
            np.load(dataset+'/npz/'+embedding)['embeddings'].astype('float32')) \
            if embedding != 'random' else None
        self.embed_size = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300

        # lstm层的设置
        self.hidden_size = 64
        self.num_layers = 1

        # 训练的参数
        self.num_epochs = 10                 # 暂时设为10
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.require_improvement = None     # 在运行的时候进行赋值


class Model(nn.Module, ABC):
    def __init__(self, config):
        """
        初始化参数
        :param config:
        """
        super(Model, self).__init__()
        # embedding层
        if config.embedding_pretrained is not None:
            self.embeddings = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False, padding_idx=config.n_vocab-1)
        else:
            self.embeddings = nn.Embedding(
                config.n_vocab, config.embed_size, padding_idx=config.n_vocab-1)

        # LSTM层
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.lstm = nn.LSTM(input_size=config.embed_size, hidden_size=config.hidden_size,
                            num_layers=config.num_layers, batch_first=True, bidirectional=True)

        # LSTM层后的全连接层
        self.hidden2tag = nn.Linear(config.hidden_size*2, config.n_tag)

        # tag_to_idx的相关设置
        self.n_tag = config.n_tag
        self.tag_to_idx = config.tag_to_idx

        # 转移矩阵
        self.transitions = nn.Parameter(
            torch.randn(config.n_tag, config.n_tag))
        self.transitions.data[self.tag_to_idx[START_TAG], :] = -10000.
        self.transitions.data[:, self.tag_to_idx[STOP_TAG]] = -10000.

        # device
        self.device = config.device

    def _get_lstm_features(self, x):
        # x: [batch_size, seq_len] -> [batch_size, seq_len, embed_size]
        x = self.embeddings(x)
        init_hidden = (torch.randn(2*self.num_layers, x.shape[0], self.hidden_size).to(self.device),
                       torch.randn(2 * self.num_layers, x.shape[0], self.hidden_size).to(self.device))
        # lstm_out: [batch_size, seq_len, hidden_size*directions]
        lstm_out, _ = self.lstm(x, init_hidden)
        # lstm_feats: [batch_size, seq_len, n_tag]
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        """
        :param feats:[batch_size, seq_len, n_tag]
        :return:
        """
        # feats:[batch_size, seq_len, n_tag]
        # 初始化α矩阵 [batch_size, n_tag]
        init_alphas = torch.full([feats.shape[0], self.n_tag], -10000.).to(self.device)
        # log空间，START_TAG有全部的权值
        init_alphas[:, self.tag_to_idx[START_TAG]] = 0.
        # 下面进行遍历
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):
            # gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2], dim=1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_idx[STOP_TAG]].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(tags.shape[0]).to(self.device)
        tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_idx[START_TAG]).long().to(self.device),
                          tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + self.transitions[tags[:, i+1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i+1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[:, -1]]
        return score

    def neg_log_likelihood(self, sentences, tags):
        feats = self._get_lstm_features(sentences)
        # feats: [batch_size, seq_len, n_tag]
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentences):
        # sentences: [batch_size, seq_len]
        lstm_feats = self._get_lstm_features(sentences)
        # lstm_feats: [batch_size, seq_len, n_tag]
        score, tag_seq = self._viterbi_decode(lstm_feats)
        # score: [batch_size], tag_seq:[batch_size, seq_len]
        return score, tag_seq

    def _viterbi_decode(self, feats):
        """
        viterbi算法求最优路径
        :param feats:
        :return:
        """
        backpointers = []
        init_vvars = torch.full((feats.shape[0], self.n_tag), -10000.).to(self.device)
        init_vvars[:, self.tag_to_idx[START_TAG]] = 0
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)
        for feat_index in range(feats.shape[1]):
            # gamr_r_l: [batch_size, n_tag, n_tag]
            gamar_r_l = torch.stack([forward_var_list[feat_index]]*feats.shape[2], dim=1)
            next_tag_var = gamar_r_l + self.transitions.unsqueeze(dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=2)  # values, idx, [batch_size, n_tag]
            # feats: [batch_size, seq_len, n_tag]
            t_r1_k = feats[:, feat_index, :]    # [batch_size, n_tag]
            forward_var_new = viterbivars_t + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(np.array(bptrs_t.tolist()))

        # [batch_size, n_tag]
        terminal_var = forward_var_list[-1] + torch.unsqueeze(self.transitions[self.tag_to_idx[STOP_TAG]], 0)
        best_tag_id = torch.argmax(terminal_var, dim=1).tolist()
        path_score = terminal_var[range(terminal_var.shape[0]), best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[range(bptrs_t.shape[0]), best_tag_id]
            best_path.append(best_tag_id.tolist())
        # best_path: [seq_len+1, batch_size]
        start = best_path.pop()
        assert start[0] == self.tag_to_idx[START_TAG]
        best_path.reverse()
        best_path = np.array(best_path).transpose().tolist()
        return path_score, best_path
