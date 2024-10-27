from typing import Callable, Type, TypeVar

import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.rl.architectures.base import BaseNetwork
from src.rl.architectures.rnn import RNNPolicyNetwork

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
