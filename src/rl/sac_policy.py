import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Кастомный извлекатель признаков для обработки сложных наблюдений.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Вызываем конструктор базового класса
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # Извлекаем размеры из пространства наблюдений
        price_data_shape = observation_space['price_data'].shape
        indicators_shape = observation_space['indicators'].shape
        common_data_shape = observation_space['common_data'].shape
        portfolio_dist_shape = observation_space['portfolio_dist'].shape

        self.portfolio_size = price_data_shape[1]
        self.time_window = price_data_shape[2]
        self.num_window_features = price_data_shape[0]
        self.num_indicators_features = indicators_shape[1]
        self.num_time_features = common_data_shape[0]
        self.portfolio_dist_size = portfolio_dist_shape[0]

        # Слои для обработки 'price_data' каждого актива
        self.price_cnn = nn.Sequential(
            nn.Conv1d(self.num_window_features, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Слои для обработки 'indicators' каждого актива
        self.indicators_fc = nn.Sequential(
            nn.Linear(self.num_indicators_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Слои для объединения признаков каждого актива
        self.asset_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Слои для обработки 'common_data'
        self.common_data_fc = nn.Sequential(
            nn.Linear(self.num_time_features, 64),
            nn.ReLU()
        )

        # Слои для обработки 'portfolio_dist'
        self.portfolio_dist_fc = nn.Sequential(
            nn.Linear(self.portfolio_dist_size, 64),
            nn.ReLU()
        )

        # Финальные слои для объединения всех признаков
        self.final_fc = nn.Sequential(
            nn.Linear(64 + 64 + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Извлекаем отдельные компоненты наблюдения
        price_data = observations['price_data']
        indicators = observations['indicators']
        common_data = observations['common_data']
        portfolio_dist = observations['portfolio_dist']

        batch_size = price_data.shape[0]

        # Обработка 'price_data' для каждого актива
        price_data = price_data.permute(0, 2, 1, 3)
        price_data = price_data.reshape(-1, self.num_window_features, self.time_window)
        price_features = self.price_cnn(price_data)
        price_features = price_features.squeeze(-1)

        # Обработка 'indicators' для каждого актива
        indicators = indicators.reshape(-1, self.num_indicators_features)
        indicators_features = self.indicators_fc(indicators)

        # Объединение признаков каждого актива
        asset_features = torch.cat([price_features, indicators_features], dim=1)
        asset_features = self.asset_fc(asset_features)
        asset_features = asset_features.view(batch_size, self.portfolio_size, -1)

        # Агрегация признаков всех активов (например, среднее значение)
        asset_features = torch.mean(asset_features, dim=1)

        # Обработка 'common_data' и 'portfolio_dist'
        common_data_features = self.common_data_fc(common_data)
        portfolio_dist_features = self.portfolio_dist_fc(portfolio_dist)

        # Объединение всех признаков
        combined_features = torch.cat([asset_features, common_data_features, portfolio_dist_features], dim=1)
        final_features = self.final_fc(combined_features)

        return final_features


class CustomSACPolicy(SACPolicy):
    """
    Кастомная политика для алгоритма SAC.
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Задаем кастомный извлекатель признаков
        super(CustomSACPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeatureExtractor,
            normalize_images=False,
            **kwargs
        )


class RNNvsCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Кастомный извлекатель признаков для обработки сложного пространства наблюдений.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Извлечение размеров входных данных
        price_data_shape = observation_space['price_data'].shape  # (N_features, portfolio_size, time_window)
        indicators_shape = observation_space['indicators'].shape  # (portfolio_size, N_indicators)
        common_data_shape = observation_space['common_data'].shape  # (N_common_features,)
        portfolio_dist_shape = observation_space['portfolio_dist'].shape  # (portfolio_size + 1,)

        # Размеры
        self.n_features = price_data_shape[0]
        self.portfolio_size = price_data_shape[1]
        self.time_window = price_data_shape[2]
        self.n_indicators = indicators_shape[1]
        self.n_common_features = common_data_shape[0]
        self.portfolio_dist_size = portfolio_dist_shape[0]

        # Слои для обработки price_data с помощью GRU
        self.price_gru_hidden_size = 64
        self.price_gru = nn.GRU(
            input_size=self.n_features * self.portfolio_size,
            hidden_size=self.price_gru_hidden_size,
            batch_first=True
        )

        # Слои для обработки indicators с помощью CNN
        self.indicators_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_indicators,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Вычисление выходной размерности для indicators_cnn
        with torch.no_grad():
            sample_indicators = torch.zeros((1, self.n_indicators, self.portfolio_size))
            sample_indicators_output = self.indicators_cnn(sample_indicators)
            indicators_output_dim = sample_indicators_output.shape[1]

        # Слои для обработки common_data
        self.common_net = nn.Sequential(
            nn.Linear(self.n_common_features, 64),
            nn.ReLU()
        )

        # Слои для обработки portfolio_dist с увеличенной емкостью
        self.portfolio_net = nn.Sequential(
            nn.Linear(self.portfolio_dist_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Общий полносвязный слой
        total_features_dim = self.price_gru_hidden_size + indicators_output_dim + 64 + 128
        self.final_net = nn.Sequential(
            nn.Linear(total_features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Обработка price_data с помощью GRU
        price_data = observations['price_data']  # Shape: (batch_size, N_features, portfolio_size, time_window)
        batch_size = price_data.shape[0]

        # Преобразование для GRU: (batch_size, time_window, N_features * portfolio_size)
        price_data = price_data.permute(0, 3, 1, 2)  # (batch_size, time_window, N_features, portfolio_size)
        price_data = price_data.reshape(batch_size, self.time_window,
                                        -1)  # (batch_size, time_window, N_features * portfolio_size)

        # Пропуск через GRU
        _, price_features = self.price_gru(price_data)  # price_features: (1, batch_size, hidden_size)
        price_features = price_features.squeeze(0)  # (batch_size, hidden_size)

        # Обработка indicators с помощью CNN
        indicators = observations['indicators']  # (batch_size, portfolio_size, N_indicators)
        indicators = indicators.permute(0, 2, 1)  # (batch_size, N_indicators, portfolio_size)
        indicators_features = self.indicators_cnn(indicators)  # (batch_size, indicators_output_dim)

        # Обработка common_data
        common_data = observations['common_data']  # (batch_size, N_common_features)
        common_features = self.common_net(common_data)  # (batch_size, 64)

        # Обработка portfolio_dist с повышенной важностью
        portfolio_dist = observations['portfolio_dist']  # (batch_size, portfolio_size + 1)
        portfolio_features = self.portfolio_net(portfolio_dist)  # (batch_size, 128)

        # Объединение всех признаков
        features = torch.cat([price_features, indicators_features, common_features, portfolio_features], dim=1)

        # Пропуск через финальный слой
        features = self.final_net(features)

        return features


class GumbelSoftmaxSACPolicy(SACPolicy):
    """
    Кастомная политика для алгоритма SAC с учетом симплекса в пространстве действий.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 features_extractor_class=RNNvsCNNFeaturesExtractor,
                 **kwargs):
        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         features_extractor_class=features_extractor_class,
                         **kwargs)

    def _predict(self, obs: dict, deterministic: bool = False) -> torch.Tensor:
        # Извлечение признаков
        logits = self.actor(obs)

        # Применяем Softmax для получения распределения
        action_prob = F.softmax(logits, dim=-1)

        # Для обеспечения стохастичности действий используем Gumbel-Softmax
        if deterministic:
            action = action_prob
        else:
            # Температура для Gumbel-Softmax
            temperature = 0.6
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(action_prob)))
            logits = (logits + gumbel_noise) / temperature
            action = F.softmax(logits, dim=-1)

        return action
