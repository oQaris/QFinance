import warnings
from typing import Type

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.rl.callbacks import EnvTerminalStatsLoggingCallback, CustomEvalCallback
from src.rl.policy import GeGLUFFNNetExtractor
from src.rl.traint_test.env_builder import build_continuous_env, load_datasets

warnings.filterwarnings("ignore",
                        message=".*To copy construct from a tensor, it is recommended to use sourceTensor.clone.*")

# Синхронизируем класс и имя агента для обучения и тестирования
agent_class: Type[BaseAlgorithm] = RecurrentPPO
exp_name = agent_class.__name__ + '_exp1'

def custom_learning_rate_schedule(remaining_progress: float, max_lr: float, min_lr: float) -> float:
    """
    Кастомное расписание для лёрнинг рейта.

    :param remaining_progress: Оставшийся прогресс обучения (от 0 до 1).
    :param max_lr: Максимальный лёрнинг рейт.
    :param min_lr: Минимальный лёрнинг рейт.
    :return: Значение лёрнинг рейта для текущего прогресса.
    """
    progress = 1 - remaining_progress
    warmup_ratio = 0.05  # Доля времени для разогрева (5% от общего времени)
    cosine_decay_ratio = 0.70  # Доля времени для косинусного спада (70% от общего времени)

    if progress < warmup_ratio:
        # Линейный рост от 0 до max_lr
        return max_lr * (progress / warmup_ratio)
    elif progress < warmup_ratio + cosine_decay_ratio:
        # Косинусное убывание от max_lr до min_lr
        cosine_progress = (progress - warmup_ratio) / cosine_decay_ratio
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * cosine_progress))
    else:
        # Поддержание min_lr
        return min_lr


def train_agent(train, test):
    print('Инициализация среды...')

    total_timesteps = 10_000_000
    num_train_envs = 1 #todo рассчитать batch для os.cpu_count()

    env_eval = build_continuous_env(test, verbose=0)
    # VecNormalize(DummyVecEnv([lambda: build_continuous_env(dataset)]))

    eval_callback = CustomEvalCallback(env_eval, best_model_save_path=f'trained_models/{exp_name}/',
                                       by_stat='sortino_ratio',
                                       deterministic=True, render=True)
    log_callback = EnvTerminalStatsLoggingCallback()

    env_train = build_continuous_env(train, verbose=0)
    # SubprocVecEnv([lambda: build_continuous_env(train) for _ in range(num_train_envs)])

    learning_rate_schedule = lambda progress: custom_learning_rate_schedule(
        progress, max_lr=3e-4, min_lr=1e-5
    )

    action_dim = env_eval.action_space.shape[0]
    features_dim = 1024
    lstm_hidden = int(features_dim / 2)
    policy_kwargs = dict(
        features_extractor_class=GeGLUFFNNetExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim, num_blocks=10, dropout=0.4),
        net_arch=[round((lstm_hidden - action_dim) / 2)],
        share_features_extractor=False,
        normalize_images=False,
        squash_output=False,
        lstm_hidden_size=lstm_hidden,
        n_lstm_layers=2,
        lstm_kwargs=dict(dropout=0.3),
        activation_fn=th.nn.Identity,
        optimizer_class=th.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4),
    )

    batch = int(256 / num_train_envs)
    agent = agent_class(
        policy='MlpLstmPolicy',
        policy_kwargs=policy_kwargs,
        n_epochs=1,
        n_steps=batch,
        # learning_starts=1,
        batch_size=batch,
        learning_rate=1e-4,
        env=env_train,
        verbose=0,
        tensorboard_log='./tensorboard_log/',
        seed=42
    )
    try:
        print('Обучение агента...')
        agent.learn(
            total_timesteps=total_timesteps,
            callback=[log_callback, eval_callback],
            progress_bar=True,
            tb_log_name=exp_name
        )
    except KeyboardInterrupt:
        print('Обучение прервано вручную. Сохраняем модель...')
    finally:
        agent.save(f'trained_models/{exp_name}/final_model')
        print('Модель успешно сохранена.')


if __name__ == '__main__':
    train_agent(*load_datasets())
