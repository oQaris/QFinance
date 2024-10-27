import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Tuple

from src.rl.architectures.base import BaseNetwork


class CNNPolicyNetwork(nn.Module, BaseNetwork):
    def __init__(
            self,
            tic_count=150,
            window_features_num=3,
            window_size=200,
            indicators_num=30,
            common_num=30,
            portfolio_size=151,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        BaseNetwork.__init__(self, last_layer_dim_pi, last_layer_dim_vf)

        # CNN для обработки исторических данных о ценах акций
        # EI3 (ensemble of identical independent inception) policy network initializer.
        # Reference article: https://doi.org/10.1145/3357384.3357961.
        k_short = 3
        k_medium = 21
        conv_mid_features = 3
        conv_final_features = 20
        n_short = window_size - k_short + 1
        n_medium = window_size - k_medium + 1
        n_long = window_size
        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=window_features_num,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
            ),
            nn.ReLU(),
        )
        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=window_features_num,
                out_channels=conv_mid_features,
                kernel_size=(1, k_medium),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
            ),
            nn.ReLU(),
        )
        self.long_term = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, n_long)),
            nn.ReLU()
        )
        self.final_convolution = nn.Conv2d(
            in_channels=2 * conv_final_features + window_features_num + 1,
            out_channels=1,
            kernel_size=(1, 1),
        )

        # Полносвязные слои для технических и фундаментальных индикаторов
        self.indicators_fc = nn.Sequential(
            nn.Linear(tic_count * indicators_num, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Полносвязные слои для общих рыночных данных
        self.common_fc = nn.Sequential(
            nn.Linear(common_num, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Полносвязный слой для распределения портфеля
        self.portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Общие слои после объединения всех признаков
        self.fc1 = nn.Linear(351, 128)

        # Actor head (политика)
        # Выход: распределение вероятностей для каждого актива и кеша
        self.actor = nn.Linear(128, self.latent_dim_pi)

        # Critic head (оценка состояния)
        # Выход: оценка ценности состояния
        self.critic = nn.Linear(128, self.latent_dim_vf)

    def calculate_common_out(self, input_dict):
        # Исторические данные о ценах акций
        # Размерность: [batch_size, window_features_num, tic_count, window_size]
        prices = input_dict['price_data']
        short_features = self.short_term(prices)
        medium_features = self.mid_term(prices)
        long_features = self.long_term(prices)

        last_stocks, cash_bias = self._process_last_action(input_dict['portfolio_dist'])
        features = torch.cat(
            [last_stocks, short_features, medium_features, long_features], dim=1
        )
        # todo
        features = self.final_convolution(features)

        # Технические и фундаментальные индикаторы
        indicators = input_dict['indicators']  # Размерность: [batch_size, tic_count, indicators_num]
        indicators = indicators.reshape(indicators.size(0), -1)  # Приведение к плоской форме
        indicators_out = self.indicators_fc(indicators)

        # Общие данные по рынку
        common = input_dict['common_data']  # Размерность: [batch_size, common_num]
        common_out = self.common_fc(common)

        # Текущее распределение портфеля
        portfolio = input_dict['portfolio_dist']  # Размерность: [batch_size, tic_count+1]
        portfolio_out = self.portfolio_fc(portfolio)

        # Объединение всех признаков
        combined = torch.cat([torch.squeeze(torch.squeeze(features, 3), 1), indicators_out, common_out, portfolio_out],
                             dim=1)
        return F.relu(self.fc1(combined))

    def forward(self, input_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of torche specified network.
            If all layers are shared, torchen ``latent_policy == latent_value``
        """
        x = self.calculate_common_out(input_dict)

        # Actor (политика)
        # action_probs = torch.softmax(self.actor(x), dim=-1)
        action_probs = self.actor(x)
        # Critic (оценка состояния)
        value = self.critic(x)

        return action_probs, value

    def forward_actor(self, input_dict) -> torch.Tensor:
        x = self.calculate_common_out(input_dict)
        return self.actor(x)

    def forward_critic(self, input_dict) -> torch.Tensor:
        x = self.calculate_common_out(input_dict)
        return self.critic(x)

    def _process_last_action(self, last_action):
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias
