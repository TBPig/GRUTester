import time

from torch import device, cuda

from utils.Comparator import save_txt
from utils.Module.MNISTModule import TorchGRU
from utils.Tester import MNISTTester

from skopt import Optimizer
from skopt.space import Integer


class MNISTFinder:
    def __init__(self):
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        print(f"device:{self.device}")

        self.data_name = "MNIST"
        self.goal = ""
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.result_path = f'result/{self.data_name}/{self.id}'

        self.info: dict[str, str] = {}
        self.tester_info: list[dict[str, str]] = []
        self.number = 0

    def run(self):
        dimensions = [
            Integer(1, 3, name='epochs'),
            Integer(512, 10240, name='hidden'),
            Integer(1, 10, name='layers')
        ]

        # 创建贝叶斯优化器
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",  # 使用高斯过程
            acq_func="EI",  # 期望改进
            acq_optimizer="auto",
            initial_point_generator="random",
            n_initial_points=8,  # 增加初始随机探索点数
            random_state=42  # 设置随机状态以确保可重现性
        )

        # 贝叶斯优化主循环
        for i in range(50):
            next_params = optimizer.ask()
            print(f"第{i + 1}次迭代: epochs={int(next_params[0])}, "
                  f"hidden={int(next_params[1])}, layers={int(next_params[2])}, "
                  , end="")
            loss = self.try_module(epochs=int(next_params[0]),
                                    hidden=int(next_params[1]),
                                    layers=int(next_params[2]))
            print(f"loss={loss:.6f}")
            optimizer.tell(next_params, loss)

            self.number += 1

        save_txt(self.info, self.tester_info, self.result_path)

    def try_module(self, epochs=10, hidden=640, layers=1):
        try:
            tester = MNISTTester(
                module=TorchGRU(hidden, layers),
                epochs=epochs,
                # 根据层数动态调整batch_size以防止内存溢出
                batch_size=max(50, 200 // layers)
            )
            tester.name += f"-{self.number:03d}"
            tester.build()

            tester.run()
            tester.save_result(self.result_path)
            self.tester_info.append(tester.get_info())

            # 返回测试损失值作为评分标准（越低越好）
            score = tester.final_loss
            del tester
            return score
        except Exception as e:
            # 如果出现内存不足等错误，返回一个较大的损失值
            print(f"出现错误: {e}")
            return 114.514  # 返回一个较大的损失值表示这次试验失败