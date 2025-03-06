from typing import Union, Callable

import gymnasium
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from torch import nn

from src.rl.callbacks import SaveVecNormalizeCallback, RecordGridVideoCallback


# Копия из rl-baselines3-zoo/utils/utils.py
def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def train_agent():
    print('Инициализация среды...')

    agent_class = RecurrentPPO
    env_build = lambda: gymnasium.make("BipedalWalker-v3", render_mode="rgb_array", hardcore=True)
    exp_name = agent_class.__name__ + '_bipedalwalker_hard'

    total_timesteps = int(10e7)
    num_train_envs = 16

    # Колбэк для записи видео
    video_callback = RecordGridVideoCallback(
        env_build,
        video_folder=f'trained_models/{exp_name}/videos',
    )
    norma_callback = SaveVecNormalizeCallback(
        save_freq=1,
        save_path=f'trained_models/{exp_name}/'
    )
    eval_callback = EvalCallback(
        VecNormalize(DummyVecEnv([env_build]), training=False, norm_reward=False),
        n_eval_episodes=20,
        eval_freq=10_000,
        best_model_save_path=f'trained_models/{exp_name}/',
        deterministic=True,
        callback_on_new_best=CallbackList([video_callback, norma_callback]),
        verbose=1
    )

    # Создаем среду для тренировки
    env_train = SubprocVecEnv([env_build] * num_train_envs)
    gamma = 0.999
    env_train = VecNormalize(env_train, gamma=gamma)

    # policy_kwargs = dict(
    #     features_extractor_class=GeGLUFFNNetExtractor,
    #     features_extractor_kwargs=dict(features_dim=128, num_blocks=5, dropout=0.1),
    #     net_arch=[64]
    # )
    policy_kwargs = dict(
        ortho_init=False,
        activation_fn=nn.ReLU,
        lstm_hidden_size=64,
        enable_critic_lstm=True,
        net_arch=dict(pi=[64], vf=[64])
    )
    batch = 256

    agent = agent_class(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env_train,
        gamma=gamma,
        n_steps=batch,
        batch_size=batch,
        ent_coef=0.001,
        learning_rate=linear_schedule(3e-4),
        clip_range=linear_schedule(0.2),
        verbose=1,
        tensorboard_log='./tensorboard_log/',
        seed=42,
        device='cpu'
    )

    try:
        print('Обучение агента...')
        agent.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback],
            tb_log_name=exp_name
        )
    except KeyboardInterrupt:
        print('Обучение прервано вручную.')
    finally:
        env_train.close()


if __name__ == '__main__':
    train_agent()
