import os
import time
from abc import ABC, abstractmethod

import torch
from torch import nn
import pandas as pd
from tqdm import tqdm


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
        if 'epoch_num' in df.columns:
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('epoch_num')))
            df = df[cols]

        df.to_csv(csv_file_path, index=False, encoding='utf-8')


class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'module'

    def set_name(self, name):
        self.name = name
        return self

    def get_info(self):
        return f"模型{self.name}"


class Tester:
    """
    测试包装类，用于封装被测试的模型以及可选的自定义训练轮次
    """

    def __init__(self, module: BasicModule, epochs: int = None):
        self.module = module
        self.epochs = epochs


class BasicComparator(ABC):
    def __init__(self):
        self.dataset_root = './data'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.data_name = "Basic"
        self.goal = "测试目的"

        self.batch_size = 1
        self.learning_rate = 2e-4
        self.epochs = 1

        self.tester_list: list[Tester] = []


    def run(self):
        infos = {"数据集": self.data_name,
                 "批大小": self.batch_size,
                 "初始学习率": self.learning_rate,
                 "测试目的": self.goal}
        model_infos = []
        self.resolve_duplicate_names()
        for tester in tqdm(self.tester_list, desc="Module List"):
            start_time = time.perf_counter()
            self._train_module(tester)
            run_time = time.perf_counter() - start_time
            model_infos.append({"模型名": tester.module.name, "模型属性": tester.module.get_info(), "时间开销": run_time})
        self.save_info(infos, model_infos)

    @abstractmethod
    def _train_module(self, tester: Tester):
        """训练单个模型"""
        pass

    @abstractmethod
    def choice(self, idx):
        """
        选择要使用的模型组
        """
        pass

    def add_tester(self, model: BasicModule, epochs: int = None):
        model = model.to(self.device)
        tester = Tester(model)
        tester.epochs = epochs if epochs else self.epochs
        self.tester_list.append(tester)
        return tester

    def save_data(self, cs: Saver, name: str):
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

    def resolve_duplicate_names(self):
        """
        检测tester_list中是否有tester.module.name是相同的，如果相同，则分别改写为***-1;***-2...的名字
        """
        # 统计每个名称出现的次数
        name_count = {}
        for tester in self.tester_list:
            name = tester.module.name
            name_count[name] = name_count.get(name, 0) + 1

        # 记录需要重命名的名称及其当前索引
        name_index = {}
        for tester in self.tester_list:
            name = tester.module.name
            if name_count[name] > 1:  # 只处理重复的名称
                if name not in name_index:
                    name_index[name] = 0  # 从第一个开始就重命名
                name_index[name] += 1
                tester.module.set_name(f"{name}-{name_index[name]}")
