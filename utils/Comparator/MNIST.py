import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utils.Comparator import module
from utils.Comparator.Basic import BasicComparator, BasicModule, Saver


class MNISTModule(BasicModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__()
        self.input_size = 28
        self.hidden_size = hidden_size
        self.output_size = 10
        self.dropout_p = dropout
        self.num_layers = num_layers

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
        base_info = f"\n模型{self.name}:hidden_size={self.hidden_size}"
        base_info += f", num_layers={self.num_layers}"
        if self.dropout_p > 0:
            base_info += f", dropout={self.dropout_p}"
        return base_info


class LocalGRU(MNISTModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "LocalGRU"
        self.gru = module.LocalGRU(self.input_size, hidden_size)


class TorchGRU(MNISTModule):
    def __init__(self, hidden_size: int, num_layers=1, batch_first=True,
                 dropout: float = 0.0):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "TorchGRU"
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0  # 只有在多层时才使用dropout
        )


class HGRU(MNISTModule):
    def __init__(self, hidden_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "HGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = module.HGRU(self.input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = super().get_info()
        base_info += f", mlp=[{self.mpl_n},{self.mpl_h}]"
        return base_info


class NormGRU(MNISTModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "normGRU"
        self.gru = module.NormGRU(self.input_size, hidden_size)


class NormHGRU(MNISTModule):
    def __init__(self, hidden_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "normHGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = module.NormHGRU(self.input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = super().get_info()
        base_info += f", mlp=[{self.mpl_n},{self.mpl_h}]"
        return base_info

class MNISTComparer(BasicComparator):

    def __init__(self):
        super().__init__()
        self.data_name = "MNIST"
        self.batch_size = 100
        self.hidden_dim = 640
        self.epochs = 30

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
            for h in [640, 800, 1024, 1536]:
                for l in range(1,5):
                    for d in [0.0,0.2]:
                        self.add_tester(TorchGRU(hidden_size=h, num_layers=l, dropout=d))
        if idx == 1:
            self.goal = "hidden层最优解-2"
            self.epochs = 80
            for h in [4096, 5120, 6144, 7168, 8192]:
                self.add_tester(TorchGRU(h).set_name(f"TorchGRU-{h}"))

        if idx == 2:
            self.goal = "测算epoch大致等于多少比较合适"
            self.add_tester(TorchGRU(hidden_size=1536,num_layers=4), epochs=250)
            self.add_tester(TorchGRU(hidden_size=1536,num_layers=4), epochs=200)
            self.add_tester(TorchGRU(hidden_size=1536,num_layers=4), epochs=150)
            self.add_tester(TorchGRU(hidden_size=1536,num_layers=4), epochs=100)
            self.add_tester(TorchGRU(hidden_size=1536,num_layers=4), epochs=50)

        if idx == 3:
            self.goal = "测算epoch大致等于多少比较合适"
            self.epochs = 80
            for h in [1024, 2048, 4096, 8192]:
                for l in range(8,12):
                    self.add_tester(TorchGRU(hidden_size=h, num_layers=l))


    def _train_module(self, tester):
        """训练单个模型"""
        module = tester.module
        epoch_num = tester.epochs
        criterion = nn.CrossEntropyLoss()
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
