import os
import time
from abc import ABC, abstractmethod

import torch
from torch import nn as nn
from tqdm import tqdm

from utils.Output import Output
from utils.SerialCounter import SerialCounter


class BasicComparator(ABC):
    """
    比较器基类，包含通用的比较逻辑
    """

    def __init__(self):
        sc = SerialCounter()
        self.serial = sc.new_serial()
        self.outputs = []

    @abstractmethod
    def run(self):
        """
        运行比较逻辑
        """
        pass

    def _save_test_text(self, data_name):
        """
        保存测试文本信息
        """
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
        info += f"数据名{data_name}"
        # 添加每个模型的名称和信息
        info += "\n模型信息：\n"
        for output in self.outputs:
            info += output.model_info + ', ' + str(round(output.consume_time, 2)) + "秒\n"

        # 追加写入文件
        with open("result/test_result.txt", "a", encoding="utf-8") as f:
            f.write(info)

    def _save_output(self, path='result/data'):
        """
        保存 outputs 列表到文件，并自动删除只保留最近的3条记录
        """
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)

        # 构建文件路径
        filename = f'{path}/{self.serial:04d}.outs'
        # 使用torch.save保存整个outputs列表
        torch.save(self.outputs, filename)

        # 获取所有保存的文件并按修改时间排序
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.outs')]
        files.sort(key=lambda x: os.path.getmtime(x))

        # 删除多余的文件，只保留最近的3个
        if len(files) > 3:
            for file_to_remove in files[:-3]:
                os.remove(file_to_remove)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'model'

    def set_name(self, name):
        self.name = name
        return self

    def get_info(self):
        return f"模型{self.name}"
