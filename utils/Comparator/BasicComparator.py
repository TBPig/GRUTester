import time

from abc import ABC, abstractmethod
from torch import device, cuda
from tqdm import tqdm

from utils.Tester import BasicTester


def save_txt(cp_info, testers_info, path):
    """保存测试文本信息"""
    info = '\n'.join([f"{key}: {value}" for key, value in cp_info.items()])

    # 添加每个模型的信息
    info += "\n模型信息：\n  "
    for info_dict in testers_info:
        info += "\n    ".join([f"{key}: {value}" for key, value in info_dict.items()])
        info += "\n  "

    with open(f"{path}/info.txt", "a", encoding="utf-8") as f:
        f.write(info)


class BasicComparator(ABC):
    def __init__(self):
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        print(f"device:{self.device}")

        self.data_name = "Basic"
        self.goal = ""
        self.id = time.strftime("%Y%m%d-%H%M%S")

        self.tester_list: list[BasicTester] = []
        self.tester_info: list[dict[str, str]] = []
        self.info: dict[str, str] = {}

    def run(self):
        self._resolve_duplicate_names()
        self.info["测试目的"] = self.goal
        result_path = f'result/{self.data_name}/{self.id}'

        for tester in tqdm(self.tester_list, desc="Module List"):
            tester.run()
            tester.save_result(result_path)
            self.tester_info.append(tester.get_info())

        save_txt(self.info, self.tester_info, result_path)

    def _resolve_duplicate_names(self):
        """
        检测tester_list中是否有tester.name是相同的，如果相同，则分别改写为***-001;***-002...的名字
        """
        # 统计每个名称出现的次数
        name_count = {}
        for tester in self.tester_list:
            name_count[tester.name] = name_count.get(tester.name, 0) + 1

        # 记录需要重命名的名称及其当前索引
        name_index = {}
        for tester in self.tester_list:
            name = tester.name
            if name_count[name] > 1:  # 只处理重复的名称
                name_index[name] = name_index.get(name, -1) + 1
                tester.name = f"{name}-{name_index[name]:03d}"

    @abstractmethod
    def choice(self, idx):
        """
        选择要使用的模型组
        """
        pass

    @abstractmethod
    def add_tester(self, model) -> BasicTester:
        """
        添加一个模型测试实例
        """
        pass
