import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.Output import Output
from utils.MPL import mpl
from utils.Comparator.Basic import BasicComparator, Model


class BasicModule(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 添加输出层
        self.fc = nn.Linear(hidden_size, self.output_size)

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.fc.weight, -init_range, init_range)
        nn.init.zeros_(self.fc.bias)

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_size}"


class GRU(BasicModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(input_size, hidden_size, output_size)
        self.name = "localGRU"

        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = nn.Linear(input_size + hidden_size, hidden_size)
        # 添加输出层
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        for linear in [self.r, self.z, self.h]:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
        nn.init.constant_(self.r.bias, -1.0)  # 使用更温和的初始化值

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        # 初始化隐藏状态
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device, x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, input_size)

            reset_update_input = torch.cat((x_t, hidden), dim=1)
            r = torch.sigmoid(self.r(reset_update_input))  # 重置门 r_t
            z = torch.sigmoid(self.z(reset_update_input))  # 更新门 z_t
            combine = torch.cat((x_t, r * hidden), dim=1)
            h_title = torch.tanh(self.h(combine))  # 候选状态 h_tilde

            # 更新隐藏状态
            hidden = (1 - z) * hidden + z * h_title
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        outputs = self.fc(outputs)  # (batch_size, seq_len, output_size)

        return outputs, hidden


class TorchGRU(BasicModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1, batch_first=True):
        super().__init__(input_size, hidden_size, output_size)
        self.name = "TorchGRU"
        # 使用 PyTorch 的 GRU 层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.init_weights()

    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)  # gru_out: (batch, seq_len, hidden_dim)
        # 应用输出层
        outputs = self.fc(gru_out)  # (batch_size, seq_len, output_size)
        return outputs, hidden


class HGRU(BasicModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, mpl_n=2, mpl_h=144):
        super().__init__(input_size, hidden_size, output_size)
        self.name = "HGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h

        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = mpl(input_size + hidden_size, hidden_size, mpl_n, mpl_h)

        self.init_weights()

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device, x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, input_size)
            combined = torch.cat((x_t, hidden), 1)

            r = torch.sigmoid(self.r(combined))  # 重置门
            z = torch.sigmoid(self.z(combined))  # 更新门
            kx_combined = torch.cat((x_t, r * hidden), 1)
            h_tilde = torch.tanh(self.h(kx_combined))  # 候选状态
            hidden = (1 - z) * hidden + z * h_tilde  # 新隐藏状态
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        outputs = self.fc(outputs)  # (batch_size, seq_len, output_size)

        return outputs, hidden

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_size},mpl=[{self.mpl_n},{self.mpl_h}]"


class NormGRU(BasicModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(input_size, hidden_size, output_size)
        self.name = "normGRU"
        self.h = nn.Linear(input_size + hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.h.weight, gain=1.0)
        nn.init.zeros_(self.h.bias)
        super().init_weights()

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device, x.dtype)
            hidden[:, 0] = 1

        # 预分配outputs张量以提高性能
        outputs = torch.empty(batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, input_size)

            combined = torch.cat((x_t, hidden), 1)
            h_tilde = self.h(combined)
            h = hidden + h_tilde
            norm = torch.norm(h, dim=1, keepdim=True)
            hidden = h / norm

            outputs[:, t, :] = hidden

        outputs = self.fc(outputs)  # (batch_size, seq_len, output_size)

        return outputs, hidden


class NormHGRU(BasicModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, mpl_n=2, mpl_h=144):
        super().__init__(input_size, hidden_size, output_size)
        self.name = "normHGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.h = mpl(input_size + hidden_size, hidden_size, mpl_n, mpl_h)
        self.init_weights()

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device, x.dtype)
            hidden[:, 0] = 1

        outputs = torch.empty(batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, input_size)
            combined = torch.cat((x_t, hidden), 1)
            h_tilde = self.h(combined)
            h = hidden + h_tilde
            norm = torch.norm(h, dim=1, keepdim=True)
            hidden = h / norm
            outputs[:, t, :] = hidden

        outputs = self.fc(outputs)  # (batch_size, seq_len, output_size)

        return outputs, hidden

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_size},mpl=[{self.mpl_n},{self.mpl_h}]"


class MNISTTester:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    dataset_root = './data'
    train_write_steps = 0.5
    test_write_steps = 1
    batch_size = 100

    def __init__(self, model):
        self.train_dataset = torchvision.datasets.MNIST(root=self.dataset_root, transform=transforms.ToTensor(),
                                                        download=True)
        self.test_dataset = torchvision.datasets.MNIST(root=self.dataset_root, transform=transforms.ToTensor(),
                                                       train=False)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True)  # 训练时打乱数据
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                                       shuffle=False)  # 测试时不需要打乱

        self.learning_rate = 1e-4
        self.batches_num = len(self.train_loader)
        # 超参数
        self.model = model.to(self.device)
        # 损失函数与优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.output = Output()

    def _prepare_data(self, input_data):
        """统一的数据预处理方法"""
        return input_data.reshape(-1, 28, 28).to(self.device)  # 输入大小=28:每行28像素；序列长度=28:28行

    def run(self, epoch_num):
        # 开始计时
        start_time = time.perf_counter()

        train_loss_sum = 0

        for i in tqdm(range(epoch_num), desc='Train', colour='white', leave=False):
            for j, (images, labels) in enumerate(self.train_loader):
                loss = self._train(images, labels)
                train_loss_sum += loss

            loss = train_loss_sum / self.batches_num
            self.output.add_train_info(loss, i + 1)
            train_loss_sum = 0

            self._test()
            cr, loss = self._test()
            self.output.add_test_info(cr, loss, i + 1)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        (self.output.set_time(run_time)
         .set_model(self.model)
         .add_info("epoch_num", epoch_num)
         .add_info("lr", self.learning_rate)
         .add_info("batch_size", self.batch_size)
         )

    def _train(self, input_data, labels):
        input_data = self._prepare_data(input_data)
        labels = labels.to(self.device)

        # 前向计算
        outputs, _ = self.model(input_data)
        predict = outputs[:, -1, :]
        loss = self.criterion(predict, labels)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _test(self):
        with torch.no_grad():
            correct = 0
            samples = 0
            test_losses = 0
            for input_data, labels in self.test_loader:
                # 同样重塑测试图像为 [N, 28, 28]
                input_data = self._prepare_data(input_data)
                labels = labels.to(self.device)
                # 前向计算
                outputs, _ = self.model(input_data)

                predicted = torch.max(outputs[:, -1, :].data, 1)[1]
                samples += labels.size(0)
                correct += (predicted == labels).sum().item()

                predict = outputs[:, -1, :]
                test_losses += self.criterion(predict, labels).item()
            correct_rate = correct / samples
            loss = test_losses / len(self.test_loader)
            return correct_rate, loss


class MNISTComparer(BasicComparator):
    data_name = "MNIST"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    input_dim = 28
    hidden_dim = 640
    output_dim = 10

    def __init__(self):
        super().__init__()
        self.models = None
        self.epoch_num = 40

    def choice(self, idx=0):
        self.models = [GRU(self.input_dim, self.hidden_dim, self.output_dim)]
        if idx == 0:
            self.models = []
            for a in range(9, 15):
                i = int(1.7 ** a)
                self.models.append(GRU(self.input_dim, i, self.output_dim))
        elif idx == 1:
            for a in range(9, 14):
                i = int(1.7 ** a)
                self.models.append(
                    HGRU(self.input_dim, self.hidden_dim, self.output_dim, mpl_h=i).set_name(f"HGRU-{i}"))
        elif idx == 2:
            for a in range(9, 15):
                i = int(1.7 ** a)
                self.models.append(
                    HGRU(self.input_dim, i, self.output_dim, mpl_h=582).set_name(f"HGRU-{i}"))
        elif idx == 3:
            for a in range(10, 15):
                i = int(1.7 ** a)
                self.models.append(
                    NormGRU(self.input_dim, i, self.output_dim).set_name(f"normGRU-{i}"))
        elif idx == 4:
            for a in range(8, 13):
                i = int(1.7 ** a)
                self.models.append(
                    NormHGRU(self.input_dim, 640, self.output_dim, mpl_h=i).set_name(f"normHGRU-{i}"))

    def run(self):
        for model in tqdm(self.models, desc="Module List"):
            t = MNISTTester(model)
            t.run(self.epoch_num)
            self.outputs.append(t.output)
        self._save_test_text(self.data_name)
        self._save_output()
