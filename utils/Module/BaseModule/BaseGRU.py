import torch
from torch import nn


class BaseGRU(nn.Module):
    """GRU模型的基类，包含所有GRU变体的共同功能"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(BaseGRU, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.dropout_p: float = dropout

        # 创建多层GRU单元
        self.gru_cells = nn.ModuleList()
        self.dropout_layers = nn.ModuleList() if dropout > 0.0 else None

        # 创建每一层的GRU单元
        for layer in range(num_layers):
            # 第一层使用input_size，其余层使用hidden_size作为输入大小
            layer_input_size = input_size if layer == 0 else hidden_size
            self.gru_cells.append(self._create_gru_cell(layer_input_size, hidden_size))

        # Dropout层（除了最后一层）
        if dropout > 0.0 and num_layers > 1:
            self.dropout_layers = nn.ModuleList([
                nn.Dropout(dropout) for _ in range(num_layers - 1)
            ])

    def _create_gru_cell(self, input_size, hidden_size):
        """
        创建单个GRU单元，由子类实现
        """
        raise NotImplementedError

    def _init_hidden(self, batch_size, device, dtype):
        """
        初始化隐藏状态，由子类实现
        """
        raise NotImplementedError

    def _forward_step(self, cell, x, hidden, layer):
        """
        单步前向传播，由子类实现
        """
        raise NotImplementedError

    def forward(self, inputs, hidden=None):
        batch_size, seq_len, input_size = inputs.size()
        if hidden is None:
            hidden = self._init_hidden(batch_size, inputs.device, inputs.dtype)

        # 逐层处理
        current_input = inputs
        # 为避免inplace操作，创建一个新的hidden张量用于存储更新后的状态
        new_hidden = hidden.clone()
        
        for layer in range(self.num_layers):
            layer_outputs = torch.empty(batch_size, seq_len, self.hidden_size, device=inputs.device, dtype=inputs.dtype)
            layer_hidden = hidden[layer]

            cell = self.gru_cells[layer]

            # 逐时间步处理
            for i in range(seq_len):
                x = current_input[:, i, :]
                layer_hidden = self._forward_step(cell, x, layer_hidden, layer)
                layer_outputs[:, i, :] = layer_hidden

            # 应用dropout（除了最后一层）
            if self.dropout_layers is not None and layer < self.num_layers - 1:
                current_input = self.dropout_layers[layer](layer_outputs)
            else:
                current_input = layer_outputs

            # 更新该层的最终隐藏状态（避免inplace操作）
            new_hidden[layer] = layer_hidden

        # outputs是最后一层的输出
        outputs = current_input

        # 修改hidden的形状为[num_layers, batch_size, 1, hidden_size]
        new_hidden = new_hidden.unsqueeze(2)
        return outputs, new_hidden