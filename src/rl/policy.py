from typing import Any, Dict, List, Optional, Type, Union
from typing import Callable, TypeVar

import gym
import torch as th
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from src.rl.architectures.base import BaseNetwork
from src.rl.architectures.geglu_ffn import GeGLUFFNNetwork
from src.rl.architectures.rnn import RNNPolicyNetwork
from src.rl.distr import DirichletDistribution

NN = TypeVar('NN', bound=BaseNetwork)


class NoneExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, 1)

    # noinspection PyMethodMayBeStatic
    def forward(self, observations):
        # Не меняем наблюдение, здесь передаётся dict, отправляем в сыром виде в модель
        return observations


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            policy_network_class: Type[NN] = RNNPolicyNetwork,
            *args,
            **kwargs,
    ):
        self.policy_network_class = policy_network_class
        # Только теперь можно вызвать super, поскольку в нём вызывается _build_mlp_extractor()
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            ortho_init=False,  # Отключаем ортогональную инициализацию
            normalize_images=False,  # Отключаем нормализацию, потому что у нас своя
            features_extractor_class=NoneExtractor
        )

    # noinspection PyUnresolvedReferences
    def _build_mlp_extractor(self) -> None:
        # Необходимо синхронизировать с BaseNetwork.extract_features_from_dict()
        portfolio_size = self.observation_space['portfolio_dist'].shape[0]
        tic_count = portfolio_size - 1
        window_features_num = self.observation_space['price_data'].shape[0]
        window_size = self.observation_space['price_data'].shape[2]
        indicators_num = self.observation_space['indicators'].shape[1]
        common_num = self.observation_space['common_data'].shape[0]
        self.mlp_extractor = self.policy_network_class(tic_count=tic_count,
                                                       window_features_num=window_features_num,
                                                       window_size=window_size,
                                                       indicators_num=indicators_num,
                                                       common_num=common_num)


class GeGLUFFNNetExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.Space,
                 features_dim: int,
                 num_blocks: int,
                 dropout: float) -> None:
        super().__init__(observation_space, features_dim)
        input_dim = 0
        if observation_space is gym.spaces.Dict:
            for key, subspace in observation_space.spaces.items():
                input_dim += get_flattened_obs_dim(subspace)
        else:
            input_dim = get_flattened_obs_dim(observation_space)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, features_dim)
        self.geglu_ffn_net = GeGLUFFNNetwork(dim=features_dim, num_blocks=num_blocks, dropout=dropout)

    def forward(self, observations) -> th.Tensor:
        if type(observations) is dict:
            tensors = []
            for key, tensor in observations.items():
                tensors.append(self.flatten(tensor))
            x = th.cat(tensors, dim=1)
        else:
            x = self.flatten(observations)
        x = self.linear(x)
        x = self.geglu_ffn_net(x)
        return x


class CustomDirichletRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms using Dirichlet distribution.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            ortho_init: bool = True,
            use_sde: bool = False,  # Should be False for Dirichlet
            log_std_init: float = 0.0,  # Not used for Dirichlet
            full_std: bool = True,  # Not used for Dirichlet
            use_expln: bool = False,  # Not used for Dirichlet
            squash_output: bool = False,  # Not relevant for Dirichlet
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            lstm_hidden_size: int = 256,
            n_lstm_layers: int = 1,
            shared_lstm: bool = False,
            enable_critic_lstm: bool = True,
            lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Ensure use_sde is False
        if use_sde:
            raise ValueError("State Dependent Exploration (use_sde) is not compatible with Dirichlet distribution.")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde=False,  # Force use_sde to False
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )

        # Dirichlet distribution
        self.action_dist = DirichletDistribution(get_action_dim(action_space))

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        alpha = self.action_net(latent_pi)  # action_net outputs alpha parameters
        return self.action_dist.proba_distribution(alpha)
