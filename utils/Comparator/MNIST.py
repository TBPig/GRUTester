import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utils.Comparator import module
from utils.Comparator.Basic import BasicComparator, BasicModule, Saver


class MNISTModule(BasicModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_size = 28
        self.hidden_size = hidden_size
        self.output_size = 10
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

    def forward(self, x):
        if self.gru is None:
            # 请先选择gru模型
            raise NotImplementedError("请先选择gru模型")
        x = x.reshape(-1, 28, 28)
        gru_out, hidden = self.gru(x)
        if self.dropout is not None:
            gru_out = self.dropout(gru_out)
        outputs = self.fc(gru_out)
        return outputs[:, -1, :]

    def get_info(self):
        base_info = f"模型{self.name}:hidden_size={self.hidden_size}"
        if self.dropout_p > 0:
            base_info += f",dropout={self.dropout_p}"
        return base_info


class LocalGRU(MNISTModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__(hidden_size, dropout)
        self.name = "LocalGRU"
        self.gru = module.LocalGRU(self.input_size, hidden_size)


class TorchGRU(MNISTModule):
    def __init__(self, hidden_size: int, num_layers=1, batch_first=True,
                 dropout: float = 0.0):
        super().__init__(hidden_size, dropout)
        self.name = "TorchGRU"
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0  # 只有在多层时才使用dropout
        )


class HGRU(MNISTModule):
    def __init__(self, hidden_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0):
        super().__init__(hidden_size, dropout)
        self.name = "HGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = module.HGRU(self.input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = f"模型{self.name}:hidden_size={self.hidden_size},mlp=[{self.mpl_n},{self.mpl_h}]"
        if self.dropout_p > 0:
            base_info += f",dropout={self.dropout_p}"
        return base_info


class NormGRU(MNISTModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__(hidden_size, dropout)
        self.name = "normGRU"
        self.gru = module.NormGRU(self.input_size, hidden_size)


class NormHGRU(MNISTModule):
    def __init__(self, hidden_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0):
        super().__init__(hidden_size, dropout)
        self.name = "normHGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = module.NormHGRU(self.input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = f"模型{self.name}:hidden_size={self.hidden_size},mlp=[{self.mpl_n},{self.mpl_h}]"
        if self.dropout_p > 0:
            base_info += f",dropout={self.dropout_p}"
        return base_info


class CNNModule(BasicModule):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.name = "CNN"
        self.input_size = 28
        self.output_size = 10
        self.dropout_p = dropout

        # 使用torchvision内置的ResNet18模型
        self.model = torchvision.models.resnet18(weights=None)  # 不使用预训练权重
        # 修改第一层以适应灰度图像（MNIST是单通道图像）
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后一层以适应10个类别（MNIST数字0-9）
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        # 初始化权重（对于最后添加的层）
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.model.fc.weight, 0, 0.01)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        """执行一次训练步骤"""
        # x shape: (batch_size, 28, 28)
        # 检查是否已经有通道维度
        if x.dim() == 3:  # (batch_size, height, width)
            x = x.unsqueeze(1)  # 添加通道维度 (batch_size, 1, 28, 28)
        elif x.dim() == 4 and x.shape[1] != 1:  # (batch_size, channels, height, width)但通道数不是1
            x = x[:, 0:1, :, :]  # 取第一个通道
        # 如果已经是 (batch_size, 1, 28, 28) 格式，则不需要改变
        x = self.model(x)
        return x

    def get_info(self):
        base_info = f"模型{self.name}"
        if self.dropout_p > 0:
            base_info += f",dropout={self.dropout_p}"
        return base_info


class MNISTComparer(BasicComparator):

    def __init__(self):
        super().__init__()
        self.data_name = "MNIST"
        self.batch_size = 100
        self.hidden_dim = 640
        self.epoch_num = 30

        # 初始化数据集和数据加载器
        self.train_dataset = torchvision.datasets.MNIST(root=self.dataset_root, transform=transforms.ToTensor(),
                                                        download=True)
        self.test_dataset = torchvision.datasets.MNIST(root=self.dataset_root, transform=transforms.ToTensor(),
                                                       train=False)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True)  # 训练时打乱数据
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                                       shuffle=False)  # 测试时不需要打乱

    def _train(self, module, criterion, optimizer):
        """执行一次训练步骤"""
        loss_sum = 0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            predicts = module(images)
            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        return loss_sum / len(self.train_loader)

    def _test(self, model, criterion):
        """测试模型"""
        with torch.no_grad():
            correct = 0
            loss_sum = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                predicts = model(images)
                loss = criterion(predicts, labels)
                predict_labels = torch.max(predicts.data, 1)[1]

                loss_sum += loss.item()
                correct += (predict_labels == labels).sum().item()

            correct_rate = correct / len(self.test_dataset)
            loss = loss_sum / len(self.test_loader)
            return correct_rate, loss

    def choice(self, idx=0):
        if idx == 0:
            self.add_tester(TorchGRU(512))
        if idx == 1:
            for h in [384, 512, 640]:
                self.add_tester(TorchGRU(h).set_name(f"TorchGRU-{h}"))
                self.add_tester(LocalGRU(h).set_name(f"LocalGRU-{h}"))


    def _train_module(self, tester):
        """训练单个模型"""
        module = tester.module
        epoch_num = tester.epochs
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            amsgrad=True
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-6)

        cs = Saver()
        for i in range(epoch_num):
            train_loss = self._train(module, criterion, optimizer)
            cr, test_loss = self._test(module, criterion)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            cs.add_epoch_data(epoch=i,
                              train_loss=train_loss,
                              test_loss=test_loss,
                              test_acc=cr,
                              learning_rate=current_lr)

        self.save_data(cs, module.name)
