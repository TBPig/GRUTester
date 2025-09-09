import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import time
from tqdm import tqdm

from utils.Output import Output
from utils.Comparator.Basic import BasicComparator, Model


class CopyTaskGenerator:
    """
    生成Copy Task数据集的类
    Copy Task要求模型记住并复制输入序列中的特定模式
    """

    def __init__(self, sequence_length: int = 10, vocab_size: int = 10):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        # 前两个token是特殊标记：0=开始标记，1=结束标记
        self.start_token = 0
        self.end_token = 1

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成一个batch的Copy Task数据
        输入序列格式: [START] + [random_sequence] + [END] + [zeros]
        输出序列格式: [zeros] + [random_sequence]
        """
        # 参数合法性检查
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 to accommodate START and END tokens.")
        if self.vocab_size <= 2:
            raise ValueError("vocab_size must be greater than 2 to allow random token generation.")

        # 计算随机序列长度
        random_seq_length = self.sequence_length - 2  # 减去start和end标记

        # 生成随机序列（不包括特殊标记）
        random_sequence = np.random.randint(2, self.vocab_size, size=(batch_size, random_seq_length))

        # 构建输入序列: START + random_sequence + END + zeros
        input_seq = np.zeros((batch_size, self.sequence_length * 2))
        start_pos = 0
        rand_start = 1
        rand_end = rand_start + random_seq_length
        end_pos = rand_end

        input_seq[:, start_pos] = self.start_token  # START标记
        input_seq[:, rand_start:rand_end] = random_sequence  # 随机序列
        input_seq[:, end_pos] = self.end_token  # END标记
        # 剩余位置默认为0（zeros填充）

        # 构建目标序列: zeros + random_sequence + zeros
        target_seq = np.zeros((batch_size, self.sequence_length * 2))
        target_seq[:, self.sequence_length:self.sequence_length + random_seq_length] = random_sequence  # 在正确位置放置随机序列

        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)


class BaseModel(Model):
    """CopyTask模型的基类"""

    def __init__(self, name: str, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_weights(self):
        """初始化权重"""
        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.uniform_(self.fc.weight, -init_range, init_range)
        nn.init.zeros_(self.fc.bias)

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_dim}"


class GRU(BaseModel):
    """自定义GRU实现"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__("localGRU", vocab_size, embedding_dim, hidden_dim)

        self.r = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.z = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.h = nn.Linear(embedding_dim + hidden_dim, hidden_dim)

        self.init_weights()
        nn.init.constant_(self.r.bias, -1.0)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, hidden), dim=1)

            r = torch.sigmoid(self.r(combined))
            z = torch.sigmoid(self.z(combined))
            kx_combined = torch.cat((x_t, r * hidden), dim=1)
            h_title = torch.tanh(self.h(kx_combined))

            hidden = (1 - z) * hidden + z * h_title
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)
        outputs = self.fc(outputs)

        return outputs, hidden


class TorchGRU(BaseModel):
    """使用PyTorch内置GRU实现"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers=1):
        super().__init__("TorchGRU", vocab_size, embedding_dim, hidden_dim)

        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.init_weights()

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            # 调整hidden形状以适应GRU层
            hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)

        outputs, hidden = self.gru(x, hidden)
        outputs = self.fc(outputs)

        return outputs, hidden


class CopyTaskTrainer:
    """
    Copy Task训练器
    """

    def __init__(self, model, sequence_length: int = 10, vocab_size: int = 10):
        self.model = model
        self.data_generator = CopyTaskGenerator(sequence_length, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充值0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(self, train_batches: int = 100, batch_size: int = 32):
        """
        训练模型
        """
        self.model.train()
        epoch_loss = 0
        for _ in range(train_batches):
            inputs, targets = self.data_generator.generate_batch(batch_size)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)

            loss = self.criterion(outputs.view(-1, self.model.vocab_size), targets.view(-1))
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / train_batches

        return avg_loss


class CopyTaskTester:
    """
    Copy Task测试器
    """

    def __init__(self, model, sequence_length: int = 10, vocab_size: int = 10):
        self.model = model
        self.data_generator = CopyTaskGenerator(sequence_length, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充值0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test(self, test_batches: int = 100, batch_size: int = 32) -> tuple:
        """
        测试模型在Copy Task上的准确率和损失
        返回: (准确率, 损失)
        """
        self.model.eval()
        self.model.to(self.device)
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for _ in range(test_batches):
                inputs, targets = self.data_generator.generate_batch(batch_size)

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, _ = self.model(inputs)
                # 计算损失
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_loss += loss.item()
                total_batches += 1

                # outputs shape: [batch_size, seq_len, vocab_size]

                # 只计算需要预测的位置（非零位置）
                mask = targets != 0
                if mask.sum() > 0:
                    predicted = torch.argmax(outputs, dim=-1)
                    correct_predictions += (predicted[mask] == targets[mask]).sum().item()
                    total_predictions += mask.sum().item()

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        return accuracy, avg_loss

    def evaluate_sequence(self, batch_size: int = 1) -> dict:
        """
        评估单个序列的复制效果，用于调试和可视化
        """
        self.model.eval()
        self.model.to(self.device)
        inputs, targets = self.data_generator.generate_batch(batch_size)

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad():
            outputs, _ = self.model(inputs)
            predicted = torch.argmax(outputs, dim=-1)

        return {
            'inputs': inputs.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'predicted': predicted.cpu().numpy(),
            'outputs': outputs.cpu().numpy()
        }


class CopyTaskComparer(BasicComparator):
    """CopyTask比较器，用于比较不同模型在CopyTask上的表现"""

    def __init__(self):
        super().__init__()
        self.sequence_length = 10
        self.vocab_size = 10
        self.embedding_dim = 32
        self.hidden_dim = 32
        self.batch_size = 32
        self.train_batches = 100
        self.test_batches = 100
        self.epoch_num = 2

        # 创建模型列表
        self.models = [
            GRU(self.vocab_size, self.embedding_dim, self.hidden_dim),
            TorchGRU(self.vocab_size, self.embedding_dim, self.hidden_dim)
        ]

    def run(self):
        """运行比较测试"""
        for model in tqdm(self.models, desc="Module List"):
            # 创建训练器和测试器
            trainer = CopyTaskTrainer(model, self.sequence_length, self.vocab_size)
            tester = CopyTaskTester(model, self.sequence_length, self.vocab_size)

            # 创建输出对象并记录结果
            output = Output()

            start_time = time.perf_counter()

            # 每轮训练后进行测试
            for epoch in tqdm(range(self.epoch_num), desc="Epoches", leave=False):
                # 训练一个epoch
                train_loss = trainer.train(self.train_batches, self.batch_size)

                # 测试模型
                accuracy, loss = tester.test(self.test_batches, self.batch_size)

                # 添加训练信息
                output.add_train_info(train_loss, epoch + 1)

                # 添加测试信息 (准确率, 损失, epoch)
                output.add_test_info(accuracy, loss, epoch + 1)

            end_time = time.perf_counter()
            run_time = end_time - start_time

            output.set_model(model)
            output.set_time(run_time)
            output.add_info("sequence_length", self.sequence_length)
            output.add_info("vocab_size", self.vocab_size)
            output.add_info("embedding_dim", self.embedding_dim)
            output.add_info("batch_size", self.batch_size)
            output.add_info("epoch_num", self.epoch_num)

            self.outputs.append(output)

        self._save_test_text("CopyTask")
        self._save_output()
