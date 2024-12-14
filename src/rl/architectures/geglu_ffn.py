import torch.nn as nn


class GeGLUFFN(nn.Module):
    def __init__(self, dim, activation=nn.GELU()):
        """
        GeGLUFFN блок.
        Args:
            dim (int): Размерность входного тензора.
            activation (nn.Module): Тип активации ('gelu' по умолчанию).
        """
        super(GeGLUFFN, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)
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
    def __init__(self, dim, dropout=0.1):
        """
        Один блок архитектуры.
        Args:
            dim (int): Размерность входного тензора.
            dropout (float): Параметр Dropout.
        """
        super(Block, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.gegluffn = GeGLUFFN(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.gegluffn(x)
        x = self.dropout(x)
        return x + residual


class GeGLUFFNNetwork(nn.Module):
    def __init__(self, dim, num_blocks):
        """
        Общая архитектура сети.
        Args:
            dim (int): Размерность входного тензора.
            num_blocks (int): Количество блоков.
        """
        super(GeGLUFFNNetwork, self).__init__()
        self.blocks = nn.ModuleList([
            Block(dim) for _ in range(num_blocks)
        ])
        self.rms_norm_out = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.rms_norm_out(x)
        return x
