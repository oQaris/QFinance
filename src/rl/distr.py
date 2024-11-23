from typing import Optional, Tuple

import torch as th
import torch.nn.functional as F
from stable_baselines3.common.distributions import Distribution, SelfDistribution
from torch import nn


class DirichletDistribution(Distribution):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.concentration = None  # Параметры распределения α
        self.distribution = None  # PyTorch Dirichlet distribution

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Создает слой для получения параметров α из латентного представления.
        """
        # Выходной слой, который формирует параметры α
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, alpha: th.Tensor) -> SelfDistribution:
        """
        Устанавливает параметры распределения (α) и создает PyTorch объект.
        """
        self.concentration = th.clamp(alpha, min=1e-15)  # Убедиться, что α > 0
        self.distribution = th.distributions.Dirichlet(self.concentration)
        return self

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Возвращает логарифм вероятности действия.
        """
        x = F.softmax(x, dim=-1)
        return self.distribution.log_prob(x)

    def entropy(self) -> Optional[th.Tensor]:
        """
        Возвращает энтропию распределения.
        """
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        """
        Возвращает случайный пример из распределения.
        """
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        """
        Возвращает режим распределения (самое вероятное значение).
        """
        # Для Dirichlet нет точного режима, если α < 1
        # Используем приближение режима как (α - 1), если α > 1
        mode = self.concentration - 1
        mode = th.clamp(mode, min=1e-15)  # Обеспечиваем неотрицательность
        return mode / mode.sum(dim=-1, keepdim=True)

    def actions_from_params(self, alpha: th.Tensor) -> th.Tensor:
        """
        Возвращает действия, сгенерированные из параметров α.
        """
        self.proba_distribution(alpha)
        return self.sample()

    def log_prob_from_params(self, alpha: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Возвращает действия и их логарифм вероятности, основанные на параметрах α.
        """
        actions = self.actions_from_params(alpha)
        log_prob = self.log_prob(actions)
        return actions, log_prob


if __name__ == '__main__':
    dist = DirichletDistribution(action_dim=3)
    alpha = th.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    dist.proba_distribution(alpha)

    samples = dist.sample()
    print("Samples:", samples)
    print("Sum:", samples.sum())

    log_prob = dist.log_prob(samples)
    print("Log Prob:", log_prob)

    mode = dist.mode()
    print("Mode:", mode)

    entropy = dist.entropy()
    print("Entropy:", entropy)
