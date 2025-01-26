from typing import Optional

import torch as th
from stable_baselines3.common.distributions import Distribution, SelfDistribution
from torch import nn


class DirichletDistribution(Distribution):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None  # PyTorch Dirichlet distribution

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Создает слой для получения параметров alpha из латентного представления.
        Гарантирует, что выходные значения будут положительными.
        """
        return nn.Sequential(
            nn.Linear(latent_dim, self.action_dim),
            nn.Softplus()
        )

    def proba_distribution(self, alpha: th.Tensor) -> SelfDistribution:
        """
        Устанавливает параметры распределения (alpha) и создает PyTorch объект.
        """
        if not th.all(alpha > 0):
            raise ValueError("All elements of alpha must be positive.")
        self.distribution = th.distributions.Dirichlet(alpha, validate_args=True)
        return self

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Возвращает логарифм вероятности действия.
        """
        # Определяем маску для padding векторов (все элементы равны 0)
        padding_mask = th.all(x == 0, dim=-1)  # type: ignore

        # Заменяем padding векторы на корректные симплексы (равномерное распределение)
        x_corrected = x.clone()
        x_corrected[padding_mask] = 1.0 / self.action_dim

        # Вычисляем исходные логарифмы вероятностей
        original_log_probs = self.distribution.log_prob(x_corrected)

        # Обнуляем log_prob для padding элементов
        log_probs = th.where(
            padding_mask,
            th.zeros_like(original_log_probs),
            original_log_probs
        )
        return log_probs

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
        return self.distribution.mode

    def actions_from_params(self, alpha: th.Tensor) -> th.Tensor:
        """
        Возвращает действия, сгенерированные из параметров alpha.
        """
        self.proba_distribution(alpha)
        return self.sample()

    def log_prob_from_params(self, alpha: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Возвращает действия и их логарифм вероятности, основанные на параметрах alpha.
        """
        actions = self.actions_from_params(alpha)
        log_prob = self.log_prob(actions)
        return actions, log_prob


def demo():
    dist = DirichletDistribution(action_dim=3)
    outs = th.tensor([-9.0, 0.0, 3.0, 4.0, 5.0])

    alpha_net = dist.proba_distribution_net(outs.shape[-1])
    alpha = alpha_net(outs)
    print("Alpha:", alpha)
    print()

    dist.proba_distribution(alpha)

    samples = dist.sample()
    print("Samples:", samples)
    print("Sum:", samples.sum())
    print()

    mode = dist.mode()
    print("Mode:", mode)
    print("Sum:", mode.sum())
    print()

    log_prob = dist.log_prob(samples)
    print("Log Prob:", log_prob)

    entropy = dist.entropy()
    print("Entropy:", entropy)


if __name__ == '__main__':
    demo()
