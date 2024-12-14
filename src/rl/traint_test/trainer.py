import os

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env
from pandas import DataFrame
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.callbacks import EnvTerminalStatsLoggingCallback
from src.rl.envs.continuous_env import PortfolioOptimizationEnv
from src.rl.envs.discrete_env import StockTradingEnv
from src.rl.loaders import split
from src.rl.policy import GeGLUFFNNetExtractor

time_window = 1
initial_amount = 500_000
fee_ratio = 0.003
env_check = True
subset_tics = None
# subset_tics = ['AFLT', 'GMKN', 'MOEX', 'TCSG', 'MAGN', 'LKOH', 'NLMK', 'OZON', 'POLY', 'SBER', 'VKCO', 'YDEX']


def load_dataset():
    dataset = pd.read_csv('C:/Users/oQaris/Desktop/Git/QFinance/data/pre/last_top_data_preprocess_norm_time.csv')
    return split(dataset, train_ratio=0.8, stratification=time_window)


def prepare_columns(columns):
    allux_cols = ['date', 'tic', 'lot', 'price']
    window_features = ['open',
                       'close',
                       'high',
                       'low',
                       'volume']
    time_features = ['vix'] + [f for f in columns if f.endswith('_sin') or f.endswith('_cos')]
    indicators = [f for f in columns if
                  f not in allux_cols and f not in window_features and f not in time_features][1:]
    return window_features, time_features, indicators


def build_continuous_env(dataset: DataFrame, verbose=1):
    window_features, time_features, indicators = prepare_columns(dataset.columns)
    return _build_env(PortfolioOptimizationEnv,
                      dataset,
                      initial_amount=initial_amount,
                      fee_ratio=fee_ratio,
                      time_window=time_window,
                      window_features=window_features,
                      time_features=time_features,
                      indicators_features=indicators,
                      verbose=verbose)


def build_discrete_env(dataset: DataFrame, verbose=1):
    _, _, indicators = prepare_columns(dataset.columns)
    return _build_env(StockTradingEnv,
                      dataset,
                      initial_amount=initial_amount,
                      comission_fee_pct=fee_ratio,
                      tech_indicator_list=indicators,
                      verbose=verbose)


def _build_env(env_class, dataset, **env_kwargs):
    if subset_tics is not None:
        dataset = dataset[dataset['tic'].isin(subset_tics)]
    env = env_class(dataset.copy(), **env_kwargs)
    if env_check:
        check_env(env)
    return env

def custom_learning_rate_schedule(remaining_progress: float, max_lr: float, min_lr: float) -> float:
    """
    Кастомное расписание для лёрнинг рейта.

    :param remaining_progress: Оставшийся прогресс обучения (от 0 до 1).
    :param max_lr: Максимальный лёрнинг рейт.
    :param min_lr: Минимальный лёрнинг рейт.
    :return: Значение лёрнинг рейта для текущего прогресса.
    """
    progress = 1 - remaining_progress
    warmup_ratio = 0.01  # Доля времени для разогрева (5% от общего времени)
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


def train_agent(dataset):
    print('Инициализация среды...')
    train, _ = split(dataset, train_ratio=0.8)

    exp_name = 'PPO_lstm_discrete'
    total_timesteps = 20_000_000
    n_validation = 10
    num_train_envs = os.cpu_count()

    env_eval = build_discrete_env(dataset)
    eval_callback = EvalCallback(env_eval, best_model_save_path=f'trained_models/{exp_name}/',
                                 n_eval_episodes=1, eval_freq=round(total_timesteps / (n_validation * num_train_envs)),
                                 deterministic=True, render=True)
    log_callback = EnvTerminalStatsLoggingCallback()

    env_train = SubprocVecEnv([lambda: build_discrete_env(train) for _ in range(num_train_envs)])

    learning_rate_schedule = lambda progress: custom_learning_rate_schedule(
        progress, max_lr=0.001, min_lr=0.001
    )

    action_dim = env_eval.action_space.shape
    features_dim = 1024
    policy_kwargs = dict(
        features_extractor_class=GeGLUFFNNetExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        share_features_extractor=False,
        net_arch=[round((features_dim - action_dim) / 2)],
        normalize_images=False,
        lstm_hidden_size=512,
        n_lstm_layers=1,
    )

    agent = RecurrentPPO(
        # policy=CustomDirichletRecurrentPolicy,
        policy='MlpLstmPolicy',
        policy_kwargs=policy_kwargs,
        # learning_rate=learning_rate_schedule,
        # policy_kwargs=dict(features_extractor_class=RNNvsCNNFeaturesExtractor),
        env=env_train,
        # replay_buffer_class = PrioritizedSequenceReplayBuffer,
        # buffer_size=500_000,
        verbose=1,
        tensorboard_log='./tensorboard_log/',
        seed=42
    )
    # agent = SAC.load('trained_models/agent_sac', env=env_train, learning_rate=3e-4)
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
    train_df, _ = load_dataset()
    # env_train = build_env(train, verbose=1)
    # env_train = build_discrete_env(train, verbose=1)
    train_agent(train_df)
