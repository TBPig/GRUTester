import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.Output import Output
from utils.SerialCounter import SerialCounter
from utils.Model import Model
from utils.MPL import mpl


class GRU(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size=None):
        super().__init__()
        self.name = "localGRU"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size else hidden_size

        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = nn.Linear(input_size + hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for linear in [self.r, self.z, self.h, self.fc]:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)

        nn.init.constant_(self.r.bias, -3.0)  # 常见选择：-1 ~ -3
        nn.init.uniform_(self.fc.weight, -init_range, init_range)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, input_size)
            combined = torch.cat((x_t, hidden), 1)

            r_t = torch.sigmoid(self.r(combined))  # 重置门
            z_t = torch.sigmoid(self.z(combined))  # 更新门
            k = r_t * hidden
            kx_combined = torch.cat((x_t, k), 1)
            h_tilde = torch.tanh(self.h(kx_combined))  # 候选状态
            hidden = (1 - z_t) * hidden + z_t * h_tilde  # 新隐藏状态
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        return outputs, hidden

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_size}"

class TorchGRU(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1, batch_first=True):
        super().__init__()
        self.name = "TorchGRU"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 使用 PyTorch 的 GRU 层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )


    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)  # gru_out: (batch, seq_len, hidden_size)

        return gru_out, hidden

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_size}"

class HGRU(Model):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layer=2, MLP_hidden_size=144):
        super().__init__()
        self.name = "HGRU"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.MLP_hidden_size = MLP_hidden_size

        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = mpl(input_size + hidden_size, hidden_size, num_layer, MLP_hidden_size)
        # 输出层（可选）
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步输入 (batch_size, input_size)
            combined = torch.cat((x_t, hidden), 1)

            r_t = torch.sigmoid(self.r(combined))  # 重置门
            z_t = torch.sigmoid(self.z(combined))  # 更新门
            k = r_t * hidden
            kx_combined = torch.cat((x_t, k), 1)
            h_tilde = torch.tanh(self.h(kx_combined))  # 候选状态
            hidden = (1 - z_t) * hidden + z_t * h_tilde  # 新隐藏状态
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        outputs = self.fc(outputs)  # (batch_size, seq_len, output_size)

        return outputs, hidden

    def get_info(self):
        return f"模型{self.name}:hidden_size={self.hidden_size},MPL_size=[{self.num_layer},{self.MLP_hidden_size}]"



class MNISTTester:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    dataset_root = './data'
    train_write_steps = 0.5
    test_write_steps = 1



    def __init__(self, model):
        # 超参数
        self.learning_rate = 0.001
        self.batch_size = 100
        self._set_loader()
        self.batches_num = len(self.train_loader)
        self._set_model(model)
        self.output = Output()

    def _set_model(self, model):
        self.model = model.to(self.device)
        # 损失函数与优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def _set_loader(self):
        dataset_func = self._get_dataset_func()
        train_dataset = dataset_func(root=self.dataset_root, train=True, transform=transforms.ToTensor(),
                                     download=True)
        test_dataset = dataset_func(root=self.dataset_root, train=False, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                        shuffle=True)  # 训练时打乱数据
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                       shuffle=False)  # 测试时不需要打乱

    def _get_dataset_func(self):
        return torchvision.datasets.MNIST

    def run(self, epoch_num):
        # 开始计时
        start_time = time.perf_counter()

        train_loss_sum = 0
        cr = 0

        for i in tqdm(range(epoch_num), desc='Train', colour='white', leave=False):
            for j, (images, labels) in enumerate(self.train_loader):
                loss = self._train(images, labels)
                train_loss_sum += loss

                # 每隔一段时间，记录时间段内train_loss平均值
                if self._is_write_time(i, j, self.train_write_steps):
                    loss = train_loss_sum / (self.train_write_steps * self.batches_num)
                    self.output.add_train_info(loss, self._get_x(i, j))
                    train_loss_sum = 0

                # 每隔一段时间，记录时间段内test_loss平均值、正确率
                if self._is_write_time(i, j, self.test_write_steps):
                    self._test()
                    cr, loss = self._test()
                    self.output.add_test_info(cr, loss, self._get_x(i, j))

        end_time = time.perf_counter()
        run_time = end_time - start_time
        (self.output.set_time(run_time)
         .set_model(self.model)
         .add_info("epoch_num", epoch_num)
         .add_info("lr", self.learning_rate)
         .add_info("batch_size", self.batch_size)
         )

    def _train(self, input_data, labels):
        input_data = self._reshape(input_data).to(self.device)  # 输入大小=28:每行28像素；序列长度=28:28行
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
                input_data = self._reshape(input_data).to(self.device)  # 输入大小=28:每行28像素；序列长度=28:28行
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

    def _reshape(self, indata):
        return indata.reshape(-1, 28, 28)

    def _is_write_time(self, epoch: int, batch: int, steps: float) -> bool:
        s = self._get_x(epoch, batch)
        return s % steps == 0

    def _get_x(self, epoch, batch):
        return epoch + (batch + 1) / self.batches_num


class MNISTComparer:
    file_name = "data"

    def __init__(self, models_func):
        sc = SerialCounter()
        self.serial = sc.new_serial()
        self.outputs = []
        self.models = self._get_models(models_func)

    def _get_models(self, models_func):
        return models_func(28,64,10)

    def run(self, epoch_num):
        for model in tqdm(self.models, desc="Module List"):
            t = MNISTTester(model)
            t.run(epoch_num)
            self.outputs.append(t.output)
        self._save_test_text()
        self._save_output()

    def _get_dataset_name(self):
        return "MNIST"

    def _save_test_text(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 生成要保存的信息
        info = f"\n\n=== 测试时间: {current_time} ===\n"
        info += f"测试序号:{self.serial}\n"
        # 添加超参数信息（从第一个输出中获取，因为测试信息相同）
        info += "超参数信息：\n"
        for key, value in self.outputs[0].test_info.items():
            info += f"{key} = {value}\t"
        # 添加数据集信息
        info += "\n数据集信息：\n"
        info += f"数据名{self._get_dataset_name()}"
        # 添加每个模型的名称和信息
        info += "\n模型信息：\n"
        for output in self.outputs:
            info += output.model_info + ', ' + str(round(output.consume_time, 2)) + "秒\n"

        # 追加写入文件
        with open("result/test_result.txt", "a", encoding="utf-8") as f:
            f.write(info)

    def _save_output(self):
        """保存 outputs 列表到文件
        此函数将 self.outputs 列表保存到指定路径的文件中，使用 torch.save 方法。
        """
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(f'result/{self.file_name}'):
            os.makedirs(f'result/{self.file_name}')

        # 构建文件路径
        filename = f'result/{self.file_name}/{self.serial:04d}.outs'
        # 使用torch.save保存整个outputs列表
        torch.save(self.outputs, filename)

    def load_data(self):
        """从指定的文件路径加载保存的Outputs列表"""
        return torch.load(f'result/{self.file_name}', weights_only=False)
