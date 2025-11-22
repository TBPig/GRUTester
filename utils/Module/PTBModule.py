from torch import nn as nn

from utils.Module import BaseModule


class PTBModule(nn.Module):
    def __init__(self, name, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers=1, dropout=0.0):
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = None
        # 只有当dropout大于0时才创建dropout层
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.fc = nn.Linear(hidden_dim, vocab_size)

        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.uniform_(self.fc.weight, -init_range, init_range)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        if self.gru is None:
            raise NotImplementedError("请先选择gru模型")

        x = self.embedding(x)
        # 只有当dropout_layer存在时才应用dropout
        if self.dropout is not None:
            x = self.dropout(x)
        outputs, hidden = self.gru(x, hidden)
        decoded = self.fc(outputs)
        return decoded, hidden

    def get_info(self):
        base_info = f"模型={self.name}| vocab_size={self.vocab_size}| embedding_dim={self.embedding_dim}"
        base_info += f"| hidden_dim={self.hidden_dim}| num_layers={self.num_layers}"
        if self.dropout_p > 0:
            base_info += f"| dropout={self.dropout_p}"
        return base_info


class TorchGRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, num_layers=1, dropout=0.0):
        super().__init__("TorchGRU", vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)


class GRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, num_layers=1, dropout=0.0):
        super().__init__("LocalGRU", vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.gru = BaseModule.LocalGRU(embedding_dim, hidden_dim)


class HGRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, mpl_h=144, num_layers=1, dropout=0.0):
        super().__init__("HGRU", vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.mpl_h = mpl_h
        self.gru = BaseModule.HGRU(embedding_dim, hidden_dim, mpl_n=2, mpl_h=mpl_h)

    def get_info(self):
        base_info = super().get_info()
        base_info += f"| mlp=[2,{self.mpl_h}]"
        return base_info


class NormHGRU(PTBModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim: int, mpl_h=144, num_layers=1, dropout=0.0):
        super().__init__("normHGRU", vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.mpl_h = mpl_h
        self.gru = BaseModule.NormHGRU(embedding_dim, hidden_dim, mpl_n=2, mpl_h=mpl_h)

    def get_info(self):
        base_info = super().get_info()
        base_info += f"| mlp=[2,{self.mpl_h}]"
        return base_info