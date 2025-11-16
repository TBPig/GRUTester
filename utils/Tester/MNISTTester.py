import time

from torch import no_grad
from torch import max as torch_max
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from utils.Tester import BasicTester


class MNISTTester(BasicTester):
    def __init__(self, module, epochs=1, batch_size=100, lr=2e-4):
        super().__init__(module, epochs, batch_size, lr)
        self.criterionFunc = lambda: CrossEntropyLoss()
        self.optimizerFunc = lambda lr: \
            (Adam(self.module.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=True))
        self.schedulerFunc = lambda opt, epochs: CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        # 初始化数据集和数据加载器
        self.data_name = "MNIST"
        self.train_dataset = MNIST(self.dataset_root, transform=ToTensor(), download=True)
        self.test_dataset = MNIST(self.dataset_root, transform=ToTensor(), train=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.final_loss = 42.0

    def build(self):
        self.criterion = self.criterionFunc()
        self.optimizer = self.optimizerFunc(self.learning_rate)
        self.scheduler = self.schedulerFunc(self.optimizer, self.epochs)

    def run(self):
        t = time.perf_counter()
        for i in range(self.epochs):
            train_loss = self._train()
            cr, test_loss = self._test()
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]

            self.result.add_epoch_data(epoch=i,
                                       train_loss=train_loss,
                                       test_loss=test_loss,
                                       test_acc=cr,
                                       learning_rate=current_lr)
            self.final_loss = test_loss
        self.consume_time = time.perf_counter() - t

    def _train(self):
        """执行一次训练步骤"""
        self.module.train()
        loss_sum = 0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            predicts = self.module(images)
            loss = self.criterion(predicts, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

        return loss_sum / len(self.train_loader)

    def _test(self):
        """测试模型"""
        self.module.eval()
        with no_grad():
            correct = 0
            loss_sum = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                predicts = self.module(images)
                loss = self.criterion(predicts, labels)
                predict_labels = torch_max(predicts.data, 1)[1]

                loss_sum += loss.item()
                correct += (predict_labels == labels).sum().item()

            correct_rate = correct / len(self.test_dataset)
            loss = loss_sum / len(self.test_loader)
            return correct_rate, loss
