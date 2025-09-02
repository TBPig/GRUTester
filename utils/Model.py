from torch import nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'model'

    def set_name(self, name):
        self.name = name
        return self

    def get_info(self):
        return f"模型{self.name}"
