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
import time
from typing import Optional

import numpy as np
import requests
import tarfile
import math

# PyTorch 相关导入
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

from utils.Output import Output
from utils.SerialCounter import SerialCounter
from utils.MPL import mpl
from utils.Comparator.Basic import BasicComparator, Model


# --- 1. 数据加载与预处理 ---

def maybe_download_and_extract(data_path):
    """
    下载并解压PTB数据集
    :param data_path: 数据集存储路径
    :return: 解压后的数据文件路径
    """
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

    return train_data, valid_data, test_data, word_size, word_to_id, id_to_word


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


class Info:
    INIT_RANGE = 0.1


class BaseModel(Model):
    def __init__(self, name, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout=0.5):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_weights_common(self):
        """初始化权重的通用部分"""
        nn.init.uniform_(self.embedding.weight, -Info.INIT_RANGE, Info.INIT_RANGE)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -Info.INIT_RANGE, Info.INIT_RANGE)

    def init_hidden(self, batch_size, device) -> torch.Tensor:
        # (num_layers, batch, hidden_dim)
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        if self.hidden_dim > 0:
            h[:, 0] = 1
        return h

    def forward(self, x, hidden):
        x = self.dropout(self.embedding(x))
        output = self.special_forward(x, hidden)
        decoded = self.fc(output)

        return decoded, hidden

    def special_forward(self, x, hidden):
        """默认实现，子类应重写此方法"""
        raise NotImplementedError("Subclasses should implement this method")

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_dim}"


class GRU(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, dropout=0.5):
        super().__init__("localGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.r = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.z = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.h = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for linear in [self.r, self.z, self.h]:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)

        nn.init.constant_(self.r.bias, -3.0)
        self.init_weights_common()

    def special_forward(self, x, hidden):
        batch_size, seq_len, embedding_dim = x.size()

        # 预分配输出张量以提高性能
        outputs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype)

        # 检查hidden维度是否匹配
        if hidden.size(0) != batch_size or hidden.size(1) != self.hidden_dim:
            raise ValueError(
                f"Hidden state dimension mismatch. Expected ({batch_size}, {self.hidden_dim}), got {hidden.size()}")

        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, embedding_dim)
            combined = torch.cat((x_t, hidden), 1)

            r_t = torch.sigmoid(self.r(combined))  # 重置门
            z_t = torch.sigmoid(self.z(combined))  # 更新门
            k = r_t * hidden
            kx_combined = torch.cat((x_t, k), 1)
            h_tilde = torch.tanh(self.h(kx_combined))  # 候选状态
            hidden = (1 - z_t) * hidden + z_t * h_tilde  # 新隐藏状态
            outputs[:, t, :] = hidden

        return outputs  # (batch_size, seq_len, hidden_dim)


class TorchGRU(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, num_layers=1, dropout=0.5):
        super().__init__("TorchGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.num_layers = num_layers

        self.model = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        self.init_weights_common()

    def special_forward(self, x, hidden):
        outputs, hidden = self.model(x, hidden)
        return outputs

    def init_hidden(self, batch_size, device) -> torch.Tensor:
        # (num_layers, batch, hidden_dim)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if self.hidden_dim > 0:
            h[:, 0] = 1
        return h


class NormGRU(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, dropout=0.5):
        super().__init__("normGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.h = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.h.weight, gain=1.0)
        nn.init.zeros_(self.h.bias)
        self.init_weights_common()

    def special_forward(self, x, hidden):
        batch_size, seq_len, embedding_dim = x.size()

        # 预分配输出张量以提高性能
        outputs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype)

        # 检查hidden维度是否匹配
        if hidden.size(0) != batch_size or hidden.size(1) != self.hidden_dim:
            raise ValueError(
                f"Hidden state dimension mismatch. Expected ({batch_size}, {self.hidden_dim}), got {hidden.size()}")

        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, embedding_dim)
            combined = torch.cat((x_t, hidden), 1)

            h_tilde = self.h(combined)
            h = hidden + h_tilde
            norm = torch.norm(h, dim=1, keepdim=True)
            hidden = h / norm
            outputs[:, t, :] = hidden

        return outputs  # (batch_size, seq_len, hidden_dim)


class HGRU(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, mpl_h, dropout=0.5):
        super().__init__("HGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.mpl_n = 2
        self.mpl_h = mpl_h

        self.r = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.z = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.h = mpl(embedding_dim + hidden_dim, hidden_dim, self.mpl_n, self.mpl_h)
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for linear in [self.r, self.z]:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)

        nn.init.constant_(self.r.bias, -3.0)
        self.init_weights_common()

    def special_forward(self, x, hidden):
        batch_size, seq_len, embedding_dim = x.size()

        # 预分配输出张量以提高性能
        outputs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype)

        # 检查hidden维度是否匹配
        if hidden.size(0) != batch_size or hidden.size(1) != self.hidden_dim:
            raise ValueError(
                f"Hidden state dimension mismatch. Expected ({batch_size}, {self.hidden_dim}), got {hidden.size()}")

        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, embedding_dim)
            combined = torch.cat((x_t, hidden), 1)

            r_t = torch.sigmoid(self.r(combined))  # 重置门
            z_t = torch.sigmoid(self.z(combined))  # 更新门
            k = r_t * hidden
            kx_combined = torch.cat((x_t, k), 1)
            h_tilde = torch.tanh(self.h(kx_combined))  # 候选状态
            hidden = (1 - z_t) * hidden + z_t * h_tilde  # 新隐藏状态
            outputs[:, t, :] = hidden

        return outputs  # (batch_size, seq_len, hidden_dim)

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_dim},MPL=[{self.mpl_n},{self.mpl_h}]"


class NormHGRU(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, mpl_h, dropout=0.5):
        super().__init__("normHGRU", vocab_size, embedding_dim, hidden_dim, dropout)
        self.input_size = vocab_size
        self.mpl_n = 2
        self.mpl_h = mpl_h

        self.h = mpl(embedding_dim + hidden_dim, hidden_dim, self.mpl_n, self.mpl_h)
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        self.init_weights_common()

    def special_forward(self, x, hidden):
        batch_size, seq_len, embedding_dim = x.size()

        # 预分配输出张量以提高性能
        outputs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype)

        # 检查hidden维度是否匹配
        if hidden.size(0) != batch_size or hidden.size(1) != self.hidden_dim:
            raise ValueError(
                f"Hidden state dimension mismatch. Expected ({batch_size}, {self.hidden_dim}), got {hidden.size()}")

        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, embedding_dim)
            combined = torch.cat((x_t, hidden), 1)

            h_tilde = self.h(combined)
            h = hidden + h_tilde
            norm = torch.norm(h, dim=1, keepdim=True)
            hidden = h / norm
            outputs[:, t, :] = hidden

        return outputs  # (batch_size, seq_len, hidden_dim)

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_dim},MPL_info=[{self.mpl_n},{self.mpl_h}]"


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

            # 处理批次
            batch_size = data.size(0)
            if hidden is None or hidden.size(1) != batch_size:
                hidden = model.init_hidden(batch_size, device)

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

        # 处理批次
        batch_size = data.size(0)
        if hidden is None or hidden.size(1) != batch_size:
            hidden = model.init_hidden(batch_size, device)

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

        # 可选：打印进度
        # if batch_idx % 200 == 0:
        #     print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')

    avg_train_loss = total_loss / total_words
    avg_train_ppl = math.exp(avg_train_loss)
    return avg_train_loss, avg_train_ppl  # 返回训练损失和困惑度


class PTBComparer(BasicComparator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "data/PTB"
    data_name = "PTB"
    sequence_length = 35
    batch_size = 20
    embedding_dim = 200
    hidden_dim = 200
    num_layers = 2
    dropout = 0.2
    learning_rate = 1

    # 数据准备
    extract_path = maybe_download_and_extract(data_path)
    train_data, valid_data, test_data, vocab_size, word_to_id, id_to_word = load_data(extract_path)
    train_dataset = PTBDataset(train_data, sequence_length)
    valid_dataset = PTBDataset(valid_data, sequence_length)
    test_dataset = PTBDataset(test_data, sequence_length)

    # 注意：对于 stateful RNN，shuffle 应该是 False
    # drop_last=False 允许最后一个不完整的批次
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    def __init__(self):
        super().__init__()
        self.epoch_num = 2
        # 生成不同隐藏层维度的GRU模型测试数组（200-1200）
        self.inner_models = [
            GRU(self.vocab_size, self.embedding_dim, hidden_dim, dropout=self.dropout)
            for hidden_dim in range(200, 1300, 200)  # 200, 400, 600, 800, 1000, 1200
        ]
        
    def run(self):
        for model in tqdm(self.inner_models, desc="Module List"):
            model = model.to(self.device)
            self.run_model(model, self.epoch_num)
        self._save_test_text(self.data_name)
        self._save_output()

    def run_model(self, model, epoch_num):
        output = Output()
        start_time = time.perf_counter()
        # --- 优化器和损失函数 ---
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        # 学习率调度器：当验证损失停止改善时，降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)

        for epoch in range(epoch_num):
            train_loss, train_ppl = train(model, self.train_loader, criterion, optimizer, self.device)
            val_loss, val_ppl, _ = evaluate(model, self.valid_loader, criterion, self.device)

            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

            print(f'Epoch: {epoch:02} | '
                  f'LR: {current_lr:.5f} | '
                  f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:7.4f} | '
                  f'Valid Loss: {val_loss:.4f} | Valid PPL: {val_ppl:7.4f}')
            output.add_train_info(train_loss, epoch + 1)
            test_loss, test_ppl, cr = evaluate(model, self.test_loader, criterion, self.device)
            output.add_test_info(cr, test_loss, epoch + 1)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        (output.set_time(run_time)
         .set_model(model)
         .add_info("epoch_num", epoch_num)
         .add_info("lr", self.learning_rate)
         .add_info("batch_size", self.batch_size)
         )
        self.outputs.append(output)
