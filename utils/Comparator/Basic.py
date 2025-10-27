import os
import time
from abc import ABC, abstractmethod

import torch
from torch import nn
import pandas as pd


class Saver:
    """
    用于保存模型训练结果的类
    记录每轮训练数据，最终保存为CSV文件
    """

    def __init__(self):
        self.training_data = []

    def add_epoch_data(self, **kwargs):
        self.training_data.append(kwargs)

    def save_results(self, path: str, name: str):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as e:
            raise OSError(f"无法创建目录 {path}: {e}")

        csv_file_path = os.path.join(path, f"{name}.csv")

        if not self.training_data:
            return

        df = pd.DataFrame(self.training_data)

        # 确保epoch列在最前面
        if 'epoch' in df.columns:
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('epoch')))
            df = df[cols]

        df.to_csv(csv_file_path, index=False, encoding='utf-8')


class BasicComparator(ABC):
    """
    比较器基类，包含通用的比较逻辑
    """

    def __init__(self):
        self.dataset_root = './data'
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.data_name = "Basic"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    @abstractmethod
    def run(self):
        """
        运行比较逻辑
        """
        pass

    @abstractmethod
    def choice(self, idx):
        """
        选择要使用的模型组
        """
        pass

    def save_data(self, cs:Saver, name:str):
        cs.save_results(f'result/{self.data_name}/{self.id}', name)

    def save_info(self, infos: dict, module_infos: list[dict]):
        """
        保存测试文本信息
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 生成要保存的信息
        info = f"\n=== 测试时间: {current_time} ===\n"
        for key, value in infos.items():
            info += f"{key}: {value}\n"

        # 添加每个模型的名称和信息
        info += "模型信息：\n"
        for module_info in module_infos:
            for key, value in module_info.items():
                info += f"{key}: {value}|"
            info += "\n"

        # 追加写入文件
        with open(f"result/{self.data_name}/{self.id}/info.txt", "a", encoding="utf-8") as f:
            f.write(info)


class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'module'

    def set_name(self, name):
        self.name = name
        return self

    def get_info(self):
        return f"模型{self.name}"
