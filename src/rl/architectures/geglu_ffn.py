import torch.nn as nn


class GeGLUFFN(nn.Module):
    def __init__(self, dim, hidden, activation=nn.GELU()):
        """
        GeGLUFFN блок.
        Args:
            dim (int): Размерность входного тензора.
            activation (nn.Module): Тип активации ('gelu' по умолчанию).
        """
        super(GeGLUFFN, self).__init__()
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(dim, hidden)
        self.linear_out = nn.Linear(hidden, dim)
        self.activation = activation

    def forward(self, x):
        # Две ветки линейных слоев
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x2 = self.activation(x2)
        # Поэлементное умножение
        out = x1 * x2
        out = self.linear_out(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, hidden, dropout):
        """
        Один блок архитектуры.
        Args:
            dim (int): Размерность входного тензора.
            hidden (int): Размерность скрытого состояния.
            dropout (float): Параметр Dropout.
        """
        super(Block, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.gegluffn = GeGLUFFN(dim, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.gegluffn(x)
        x = self.dropout(x)
        return x + residual


class GeGLUFFNNetwork(nn.Module):
    def __init__(self, dim, num_blocks=8, dropout=0.3, hidden_ratio=2):
        """
        Общая архитектура сети.
        Args:
            dim (int): Размерность входного тензора.
            num_blocks (int): Количество блоков.
        """
        super(GeGLUFFNNetwork, self).__init__()
        hidden = round(dim * hidden_ratio)
        self.blocks = nn.ModuleList([
            Block(dim, hidden, dropout) for _ in range(num_blocks)
        ])
        self.layer_norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm_out(x)
        return x
