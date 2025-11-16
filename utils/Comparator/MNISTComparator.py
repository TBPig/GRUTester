from utils.Comparator.BasicComparator import BasicComparator
from utils.Module.MNISTModule import TorchGRU
from utils.Tester.MNISTTester import MNISTTester


class MNISTComparer(BasicComparator):
    def __init__(self):
        super().__init__()
        self.data_name = "MNIST"

    def choice(self, idx=0):
        if idx == 0:
            self.goal = "测试新代码能不能运行"
            self.add_tester(TorchGRU(640)).set_epochs(1).build()
        if idx == 1:
            self.goal = "hidden层最优解-2"
            for h in [4096, 5120, 6144, 7168, 8192]:
                self.add_tester(TorchGRU(h).set_name(f"TorchGRU-{h}")).set_epochs(80).build()

        if idx == 2:
            self.goal = "测算epoch大致等于多少比较合适"
            self.add_tester(TorchGRU(hidden_size=1536, num_layers=4)).set_epochs(250).build()
            self.add_tester(TorchGRU(hidden_size=1536, num_layers=4)).set_epochs(200).build()
            self.add_tester(TorchGRU(hidden_size=1536, num_layers=4)).set_epochs(150).build()
            self.add_tester(TorchGRU(hidden_size=1536, num_layers=4)).set_epochs(100).build()

        if idx == 3:
            self.goal = "测算epoch大致等于多少比较合适"
            for h in [1024, 2048, 4096, 8192]:
                for l in range(8, 12):
                    self.add_tester(TorchGRU(hidden_size=h, num_layers=l)).set_epochs(80).build()

    def add_tester(self, model):
        tester = MNISTTester(model, epochs=40)
        self.tester_list.append(tester)
        return tester
