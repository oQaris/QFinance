from typing import Optional, Dict, Any, Union, Type, List, Tuple

import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.distributions import Distribution, SelfDistribution
from stable_baselines3.common.policies import ContinuousCritic, BasePolicy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, get_actor_critic_arch
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import SACPolicy, Actor
from torch.distributions import Dirichlet


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Кастомный извлекатель признаков для обработки сложных наблюдений.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Вызываем конструктор базового класса
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)

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
            features_extractor_class=CNNFeatureExtractor,
            normalize_images=False,
            **kwargs
        )


class RNNvsCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Кастомный извлекатель признаков для обработки сложного пространства наблюдений.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
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
        self.price_gru_hidden_size = 128
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

        # Общий полносвязный слой
        total_features_dim = self.price_gru_hidden_size + indicators_output_dim + self.n_common_features + self.portfolio_dist_size
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
        price_data = price_data.reshape(batch_size, self.time_window, -1)  # (batch_size, time_window, N_features * portfolio_size)

        # Пропуск через GRU
        _, price_features = self.price_gru(price_data)  # price_features: (1, batch_size, hidden_size)
        price_features = price_features.squeeze(0)  # (batch_size, hidden_size)

        # Обработка indicators с помощью CNN
        indicators = observations['indicators']  # (batch_size, portfolio_size, N_indicators)
        indicators = indicators.permute(0, 2, 1)  # (batch_size, N_indicators, portfolio_size)
        indicators_features = self.indicators_cnn(indicators)  # (batch_size, indicators_output_dim)

        common_data = observations['common_data']  # (batch_size, N_common_features)
        portfolio_dist = observations['portfolio_dist']  # (batch_size, portfolio_size + 1)

        # Объединение всех признаков
        features = torch.cat([price_features, indicators_features, common_data, portfolio_dist], dim=1)

        # Пропуск через финальный слой
        return self.final_net(features)


# Реализация распределения Дирихле
class DirichletDistribution(Distribution):
    def __init__(self, concentration: torch.Tensor):
        super().__init__()
        self.concentration = concentration
        self.distribution = Dirichlet(concentration)

    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        return None

    def proba_distribution(self: SelfDistribution, *args, **kwargs) -> SelfDistribution:
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        # Возвращаем среднее, так как мода может быть неопределена
        return self.distribution.mean

    def actions_from_params(self, concentration: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.distribution = Dirichlet(concentration)
        if deterministic:
            return self.mode()
        else:
            return self.sample()

    def log_prob_from_params(self, concentration: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.distribution = Dirichlet(concentration)
        return self.log_prob(actions)


class DummyExtractor(nn.Module):
    def __init__(self, feature_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim_pi = feature_dim
        self.latent_dim_vf = feature_dim

    def forward(self, input_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return input_dict, input_dict

    def forward_actor(self, input_dict) -> torch.Tensor:
        return input_dict

    def forward_critic(self, input_dict) -> torch.Tensor:
        return input_dict


# Кастомная политика
class CustomDirichletPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch: Optional[List[Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(CustomDirichletPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=RNNvsCNNFeaturesExtractor,
            *args,
            **kwargs,
        )

        action_dim = self.action_space.shape[0]

        # Сеть для параметров концентрации распределения Дирихле
        self.action_net = nn.Sequential(
            nn.Linear(self.features_dim, action_dim),
            nn.Softplus()  # Чтобы гарантировать положительные параметры концентрации
        )

        # Сеть для оценки ценности состояния
        self.value_net = nn.Linear(self.features_dim, 1)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DummyExtractor(self.features_dim)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        concentration = self.action_net(latent_pi) + 1e-3  # Добавляем небольшое значение, чтобы избежать нулей
        return DirichletDistribution(concentration)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(features)

        distribution = self._get_action_dist_from_latent(features)
        actions = distribution.actions_from_params(distribution.concentration, deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions


class GumbelSoftmaxSACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = True,
            log_std_init: float = -3,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = CNNFeatureExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = False,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if
                                 "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode
