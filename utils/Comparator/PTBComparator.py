from utils.Comparator.BasicComparator import BasicComparator
from utils.Module.PTBModule import TorchGRU
from utils.Tester.PTBTester import PTBTester


class PTBComparer(BasicComparator):
    def __init__(self):
        super().__init__()
        self.data_name = "PTB"

        self.embedding_dim = 200
        self.hidden_dim = 200

        self.batch_size = 50
        self.learning_rate = 1e-3
        self.sequence_length = 35

    def choice(self, idx):
        if idx == 0:
            self.goal = "测试新代码能不能运行"
            (self.add_tester(
                TorchGRU(vocab_size=10000, embedding_dim=self.embedding_dim, hidden_dim=1024, num_layers=2)
            ).set_epochs(1).build())
        elif idx == 1:
            self.goal = "不同层数对模型性能的影响"
            for layers in [2, 3, 4]:
                self.add_tester(
                    TorchGRU(vocab_size=10000, embedding_dim=self.embedding_dim, hidden_dim=1024, num_layers=layers)
                ).set_epochs(20).build()

    def add_tester(self, model) -> PTBTester:
        tester = PTBTester(model, epochs=40, batch_size=self.batch_size, lr=self.learning_rate,
                           sequence_length=self.sequence_length)
        self.tester_list.append(tester)
        return tester
