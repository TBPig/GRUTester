import os

import pandas as pd
from torch import device, cuda

from abc import ABC, abstractmethod


class Result:
    """
    用于保存模型训练结果的类
    记录每轮训练数据，最终保存为CSV文件
    """

    def __init__(self):
        self.training_data = []

    def add_epoch_data(self, **kwargs):
        self.training_data.append(kwargs)

    def save_result(self, path: str, name: str):
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
        if 'epoch_num' in df.columns:
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('epoch_num')))
            df = df[cols]

        df.to_csv(csv_file_path, index=False, encoding='utf-8')


class BasicTester(ABC):
    """
    测试包装类，用于封装被测试的模型以及可选的自定义训练轮次
    """

    def __init__(
            self,
            module,
            epochs: int = 1,
            batch_size: int = 100,
            learning_rate: float = 2e-4
    ):
        self.dataset_root = './data'
        self.data_name = ""
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        self.result = Result()

        self.module = module.to(self.device)
        self.name = module.name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.scheduler = None
        self.optimizer = None
        self.criterion = None

        self.consume_time = 0

    def set_epochs(self, epochs):
        self.epochs = epochs
        return self

    def save_result(self, path):
        self.result.save_result(path, self.name)

    def get_info(self):
        super_property = f"epochs={self.epochs}| batch={self.batch_size}| lr={self.learning_rate}| time={self.consume_time}s"
        return {"名称": self.name, "超参数": super_property, "模型属性": self.module.get_info()}

    @abstractmethod
    def build(self):
        """到此完成配置"""
        pass

    @abstractmethod
    def run(self):
        """给定超参、模型，开始一整套训练"""
        pass
