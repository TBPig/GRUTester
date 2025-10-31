# -*- coding: utf-8 -*-
"""
使用PTB数据集和PyTorch训练一个简单的循环神经网络(RNN)语言模型。

此脚本演示了如何：
1.  加载和预处理PTB数据集。
2.  构建一个基于LSTM的简单语言模型。
3.  使用PyTorch进行训练。
4.  评估模型的困惑度(Perplexity)。

注意：此示例假设PTB数据集文件 'ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt'
      已存在于当前工作目录或指定的 'data_path' 中。
      你可以从 https://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz 下载。
"""

import collections
import os
from typing import Optional
import numpy as np
import requests
import tarfile
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.Comparator.Basic import BasicComparator, BasicModule, Saver
from utils.Comparator import module


# --- 1. 数据加载与预处理 ---

def get_dataset():
    """
    下载并解压PTB数据集
    :return: 解压后的数据文件路径
    """

    data_path = "data/PTB"
    # 检查数据路径是否存在，不存在则创建
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_path = os.path.join(data_path, "simple-examples.tgz")

    # 检查压缩文件是否存在，不存在则下载
    if not os.path.exists(file_path):
        print("正在下载PTB数据集...")
        url = "https://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("下载完成。")

    # 检查解压路径是否存在，不存在则解压
    extract_path = os.path.join(data_path, "simple-examples", "data")
    if not os.path.exists(extract_path):
        print("正在解压数据集...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=data_path)
        print("解压完成。")
    return extract_path


def load_data(data_path):
    """加载并预处理PTB数据"""

    def build_vocab(file_path):
        """从文件构建词汇表"""
        with open(file_path, 'r') as f:
            data = f.read().replace('\n', '<eos>').split()
        counter = collections.Counter(data)
        # 按频率排序，最常见的在前
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = zip(*count_pairs)
        word_to_id = dict(zip(words, range(len(words))))
        id_to_word = dict((v, k) for k, v in word_to_id.items())
        return word_to_id, id_to_word

    def file_to_word_ids(file_path, word_to_id):
        """将文件中的单词转换为ID序列"""
        with open(file_path, 'r') as f:
            data = f.read().replace('\n', '<eos>').split()
        # 对于不在词汇表中的词，可以映射到一个特殊标记，这里简单忽略或使用<unk>
        # 但PTB数据集通常已经过滤，这里假设所有词都在词汇表中
        return [word_to_id[word] for word in data if word in word_to_id]

    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    word_to_id, id_to_word = build_vocab(train_path)
    word_size = len(word_to_id)

    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, word_size


class PTBDataset(Dataset):
    """用于PTB数据的PyTorch Dataset
    该数据集将输入数据分割成固定长度的序列块，每个序列的输入和目标是时间上连续的词序列，
    目标序列相对于输入序列向后偏移一个时间步，用于语言模型的训练。
    """

    def __init__(self, data, sequence_length):
        """
        :param data: 输入的数据列表，通常是一个词索引序列
        :param sequence_length: 每个样本序列的长度
        """
        self.data = np.array(data, dtype=np.int64)
        self.sequence_length = sequence_length

    def __len__(self):
        """
        返回数据集中的样本数量
        - 减1是因为需要为目标序列保留一个额外的时间步。
        :return: int: 数据集中的样本数量
        """
        return (len(self.data) - 1) // self.sequence_length

    def __getitem__(self, idx):
        """获取指定索引的样本

        根据索引计算对应的数据块位置，提取输入序列和目标序列。目标序列是输入序列
        向后偏移一个时间步的版本，用于语言模型的下一个词预测任务。

        :param idx: 样本索引
        :return: tuple: 包含两个元素的元组
                - inputs (torch.Tensor): 输入序列张量，类型为torch.long
                - targets (torch.Tensor): 目标序列张量，类型为torch.long
        """
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        inputs = self.data[start_idx:end_idx]
        targets = self.data[start_idx + 1:end_idx + 1]
        # 如果targets不够长（最后一个块），用序列的最后一个词填充目标（但这在PTB中通常不会发生，因为我们按sequence_length分块）
        # 更稳健的做法是丢弃不完整的最后一个序列，或者在__len__中处理
        # 这里我们假设数据长度合适
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class PTBModule(BasicModule):
    def __init__(self, name, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout=0.0):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = None
        # 只有当dropout大于0时才创建dropout层
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.fc = nn.Linear(hidden_dim, vocab_size)

        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.uniform_(self.fc.weight, -init_range, init_range)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden):
        if self.gru is None:
            raise NotImplementedError("请先选择gru模型")

        x = self.embedding(x)
        # 只有当dropout_layer存在时才应用dropout
        if self.dropout is not None:
            x = self.dropout(x)
        outputs, _ = self.gru(x, hidden)
        decoded = self.fc(outputs)
        return decoded, hidden

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_dim}"


class TorchGRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, num_layers=1, dropout=0.0):
        super().__init__("TorchGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)


class GRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, dropout=0.0):
        super().__init__("LocalGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.gru = module.LocalGRU(embedding_dim, hidden_dim)


class HGRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, mpl_h=144, dropout=0.0):
        super().__init__("HGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.mpl_h = mpl_h
        self.gru = module.HGRU(embedding_dim, hidden_dim, mpl_n=2, mpl_h=mpl_h)

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_dim},MPL=[2,{self.mpl_h}]"


class NormHGRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, mpl_h=144, dropout=0.0):
        super().__init__("normHGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.mpl_h = mpl_h
        self.gru = module.NormHGRU(embedding_dim, hidden_dim, mpl_n=2, mpl_h=mpl_h)

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_dim},MPL=[2,{self.mpl_h}]"


def repackage_hidden(h):
    """将隐藏状态从计算图中分离，以防止反向传播穿过整个训练历史。
    这对于 stateful RNN 非常重要。
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()  # 设置为评估模式
    total_loss = 0.
    total_words = 0
    # 注意：这里我们不能依赖 data_loader.batch_size，因为它可能因为 drop_last=False 而变化
    # 我们在循环内部获取实际的 batch_size
    hidden: Optional[torch.Tensor] = None  # 在循环开始时初始化
    correct = 0

    with torch.no_grad():  # 禁用梯度计算
        for i, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)

            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)

            # output: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch * seq_len)
            pred_target = output.view(-1, model.fc.out_features)
            loss = criterion(pred_target, targets.view(-1))
            total_loss += loss.item() * targets.numel()  # numel() gives total number of elements
            total_words += targets.numel()

            predicted = pred_target.argmax(dim=1)
            correct += (predicted == targets.view(-1)).sum().item()

    correct_rate = correct / total_words
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, correct_rate


def train(model, train_loader, criterion, optimizer, device):
    """训练模型 (移除了 scheduler)"""
    model.train()  # 设置为训练模式
    total_loss = 0.
    total_words = 0
    hidden: Optional[torch.Tensor] = None  # 在循环开始时初始化

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()  # 清零梯度
        output, hidden = model(data, hidden)

        # 分离隐藏状态，防止反向传播穿过整个历史
        hidden = repackage_hidden(hidden)

        # 计算损失
        loss = criterion(output.view(-1, model.fc.out_features), targets.view(-1))

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸 (RNN中常用)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # 更新参数
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_words += targets.numel()

    avg_train_loss = total_loss / total_words
    avg_train_ppl = math.exp(avg_train_loss)
    return avg_train_loss, avg_train_ppl  # 返回训练损失和困惑度


class PTBComparer(BasicComparator):

    def __init__(self):
        super().__init__()
        self.data_name = "PTB"
        self.sequence_length = 35
        self.batch_size = 20
        self.embedding_dim = 200
        self.hidden_dim = 200
        self.dropout = 0.0
        self.learning_rate = 1e-3
        # 数据准备
        self.extract_path = get_dataset()
        self.train_data, self.valid_data, self.test_data, self.vocab_size = load_data(self.extract_path)
        self.train_dataset = PTBDataset(self.train_data, self.sequence_length)
        self.valid_dataset = PTBDataset(self.valid_data, self.sequence_length)
        self.test_dataset = PTBDataset(self.test_data, self.sequence_length)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def choice(self, idx):
        if idx == 0:
            self.add_tester(GRU(self.vocab_size, self.embedding_dim, self.hidden_dim, self.dropout))
            pass


    def _train_module(self, tester):
        module = tester.module.to(self.device)
        epoch_num = tester.epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-6)

        cs = Saver()
        for i in range(epoch_num):
            train_loss, train_ppl = train(module, self.train_loader, criterion, optimizer, self.device)
            test_loss, test_ppl, cr = evaluate(module, self.test_loader, criterion, self.device)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            cs.add_epoch_data(epoch=i,
                              train_loss=train_loss,
                              test_loss=test_loss,
                              test_acc=cr,
                              learning_rate=current_lr)
        self.save_data(cs, module.name)
