import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.Comparator import module
from utils.Comparator.Basic import BasicComparator, BasicModule, Saver


class MNISTModule(BasicModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout

        # 添加输出层
        self.gru = None
        self.fc = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        init_range = 0.1
        nn.init.uniform_(self.fc.weight, -init_range, init_range)
        nn.init.zeros_(self.fc.bias)

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x, hidden=None):
        if self.gru is None:
            # 请先选择gru模型
            raise NotImplementedError("请先选择gru模型")

        gru_out, hidden = self.gru(x, hidden)
        if self.dropout is not None:
            gru_out = self.dropout(gru_out)
        outputs = self.fc(gru_out)
        return outputs, hidden

    def get_info(self):
        base_info = f"模型{self.name}:hidden_size={self.hidden_size}"
        if self.dropout_p > 0:
            base_info += f",dropout={self.dropout_p}"
        return base_info


class LocalGRU(MNISTModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, output_size, dropout)
        self.name = "LocalGRU"
        self.gru = module.LocalGRU(input_size, hidden_size)


class TorchGRU(MNISTModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1, batch_first=True,
                 dropout: float = 0.0):
        super().__init__(input_size, hidden_size, output_size, dropout)
        self.name = "TorchGRU"
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0  # 只有在多层时才使用dropout
        )


class HGRU(MNISTModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, output_size, dropout)
        self.name = "HGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = module.HGRU(input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = f"模型{self.name}:hidden_size={self.hidden_size},mlp=[{self.mpl_n},{self.mpl_h}]"
        if self.dropout_p > 0:
            base_info += f",dropout={self.dropout_p}"
        return base_info


class NormGRU(MNISTModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, output_size, dropout)
        self.name = "normGRU"
        self.gru = module.NormGRU(input_size, hidden_size)


class NormHGRU(MNISTModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, output_size, dropout)
        self.name = "normHGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = module.NormHGRU(input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = f"模型{self.name}:hidden_size={self.hidden_size},mlp=[{self.mpl_n},{self.mpl_h}]"
        if self.dropout_p > 0:
            base_info += f",dropout={self.dropout_p}"
        return base_info


class MNISTComparer(BasicComparator):
    dataset_root = './data'
    batch_size = 100
    learning_rate = 1e-4

    input_dim = 28
    hidden_dim = 640
    output_dim = 10
    dropout = 0.3  # 添加dropout参数

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
        self.data_name = "MNIST"
        self.models: list[MNISTModule] = []
        self.epoch_num = 40

        # 初始化数据集和数据加载器
        self.train_dataset = torchvision.datasets.MNIST(root=self.dataset_root, transform=transforms.ToTensor(),
                                                        download=True)
        self.test_dataset = torchvision.datasets.MNIST(root=self.dataset_root, transform=transforms.ToTensor(),
                                                       train=False)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True)  # 训练时打乱数据
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                                       shuffle=False)  # 测试时不需要打乱
        self.batches_num = len(self.train_loader)

    def _train_module(self, module: BasicModule, epoch_num: int):
        """训练单个模型"""
        cs = Saver()

        module = module.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)

        for i in range(epoch_num):
            train_loss = self._train(module, criterion, optimizer)
            cr, test_loss = self._test(module, criterion)
            cs.add_epoch_data(epoch=i, train_loss=train_loss, test_loss=test_loss, test_acc=cr)
        self.save_data(cs, module.name)

    def _train(self, module, criterion, optimizer):
        """执行一次训练步骤"""
        train_loss_sum = 0
        for images, labels in self.train_loader:
            input_data = images.reshape(-1, 28, 28).to(self.device)
            labels = labels.to(self.device)

            # 前向计算
            outputs, _ = module(input_data)
            predict = outputs[:, -1, :]
            loss = criterion(predict, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        return train_loss_sum / self.batches_num

    def _test(self, model, criterion):
        """测试模型"""
        with torch.no_grad():
            correct = 0
            samples = 0
            test_losses = 0
            for input_data, labels in self.test_loader:
                # 同样重塑测试图像为 [N, 28, 28]
                input_data = input_data.reshape(-1, 28, 28).to(self.device)
                labels = labels.to(self.device)
                # 前向计算
                outputs, _ = model(input_data)

                predicted = torch.max(outputs[:, -1, :].data, 1)[1]
                samples += labels.size(0)
                correct += (predicted == labels).sum().item()

                predict = outputs[:, -1, :]
                test_losses += criterion(predict, labels).item()
            correct_rate = correct / samples
            loss = test_losses / len(self.test_loader)
            return correct_rate, loss

    def choice(self, idx=0):
        self.models: list[MNISTModule] = [
            TorchGRU(self.input_dim, self.hidden_dim, self.output_dim, dropout=self.dropout)]
        if idx == 0:
            self.models = [
                TorchGRU(self.input_dim, self.hidden_dim, self.output_dim, dropout=self.dropout),
                LocalGRU(self.input_dim, self.hidden_dim, self.output_dim, dropout=self.dropout)
            ]
            pass
        elif idx == 1:
            self.models = []
            for a in range(9, 15):
                i = int(1.7 ** a)
                self.models.append(TorchGRU(self.input_dim, i, self.output_dim, dropout=self.dropout))
        elif idx == 2:
            for a in range(9, 14):
                i = int(1.7 ** a)
                self.models.append(
                    HGRU(self.input_dim, self.hidden_dim, self.output_dim, mpl_h=i, dropout=self.dropout).set_name(
                        f"HGRU-{i}"))
        elif idx == 3:
            for a in range(9, 15):
                i = int(1.7 ** a)
                self.models.append(
                    HGRU(self.input_dim, i, self.output_dim, mpl_h=582, dropout=self.dropout).set_name(f"HGRU-{i}"))
        elif idx == 4:
            for a in range(10, 15):
                i = int(2 ** a)
                self.models.append(
                    NormGRU(self.input_dim, i, self.output_dim, dropout=self.dropout).set_name(f"normGRU-{i}"))
        elif idx == 5:
            for a in range(8, 16):
                i = int(2 ** a)
                self.models.append(
                    NormHGRU(self.input_dim, 640, self.output_dim, mpl_h=i, dropout=self.dropout).set_name(
                        f"normHGRU-{i}"))

    def run(self):
        infos = {"数据集": self.data_name, "批大小": self.batch_size, "初始学习率": self.learning_rate,
                 "Dropout": self.dropout}
        model_infos = []
        for model in tqdm(self.models, desc="Module List"):
            # 开始计时
            start_time = time.perf_counter()
            self._train_module(model, self.epoch_num)
            end_time = time.perf_counter()
            model_infos.append({"模型名": model.name, "模型属性": model.get_info(), "时间开销": end_time - start_time})
        self.save_info(infos, model_infos)
