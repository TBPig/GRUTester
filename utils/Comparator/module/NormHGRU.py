import torch
from torch import nn

from utils.MLP import mlp


class NormHGRU(nn.Module):
    def __init__(self, input_size, hidden_size, mpl_n=2, mpl_h=144):
        super(NormHGRU, self).__init__()
        self.hidden_size:int = hidden_size
        self.h = mlp(input_size + hidden_size, hidden_size, mpl_n, mpl_h)

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        hidden[:, 0] = 1
        return hidden

    def forward(self, inputs, hidden=None):
        batch_size, seq_len, input_size = inputs.size()
        if hidden is None:
            hidden = self._init_hidden(batch_size, inputs.device, inputs.dtype)

        # 预分配outputs张量以提高性能
        outputs = torch.empty(batch_size, seq_len, self.hidden_size, device=inputs.device, dtype=inputs.dtype)

        for i in range(seq_len):
            x = inputs[:, i, :]

            # 计算候选隐藏状态
            combined = torch.cat([x, hidden], dim=1)
            h_tilde = self.h(combined)

            # 更新隐藏状态并标准化
            h = hidden + h_tilde
            norm = torch.norm(h, dim=1, keepdim=True)
            hidden = h / norm

            outputs[:, i, :] = hidden

        return outputs, hidden