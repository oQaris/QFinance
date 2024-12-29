import warnings

import numpy as np
import pandas as pd
import torch as th
from gymnasium.utils.env_checker import check_env
from pandas import DataFrame
from stable_baselines3 import SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.callbacks import EnvTerminalStatsLoggingCallback, CustomEvalCallback
from src.rl.envs.continuous_env import PortfolioOptimizationEnv
from src.rl.envs.discrete_env import StockTradingEnv
from src.rl.loaders import split
from src.rl.policy import GeGLUFFNNetExtractor

warnings.filterwarnings("ignore",
                        message=".*For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]).*")
warnings.filterwarnings("ignore",
                        message=".*A Box observation space has an unconventional shape (neither an image, nor a 1D vector).*")
warnings.filterwarnings("ignore",
                        message=".*It seems a Box observation space is an image but the lower and upper bounds are not [0, 255].*")
warnings.filterwarnings("ignore",
                        message=".*A Box observation space minimum value is -infinity. This is probably too low.*")
warnings.filterwarnings("ignore",
                        message=".*A Box observation space maximum value is -infinity. This is probably too high.*")
warnings.filterwarnings("ignore",
                        message=".*Not able to test alternative render modes due to the environment not having a spec.*")

time_window = 1
initial_amount = 500_000
fee_ratio = 0.003
env_check = True
subset_tics = None
# subset_tics = ['AFLT', 'GMKN', 'MOEX', 'TCSG', 'MAGN', 'LKOH', 'NLMK', 'OZON', 'POLY', 'SBER', 'VKCO', 'YDEX']


def load_dataset():
    # todo 2024-01-04_2024-10-04_5_MIN_final
    dataset = pd.read_csv('../../../data/pre/2018-01-01_2024-12-21_HOUR_final.csv')
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['lot'] = dataset['lot'].astype('int32')
    dataset = dataset.astype({col: 'float32'
                              for col in dataset.select_dtypes(include=['float64']).columns
                              if col != 'price'})
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

    exp_name = 'SAC_lstm_continuous_exp'
    total_timesteps = 10_000_000
    num_train_envs = 1 #todo рассчитать batch для os.cpu_count()

    env_eval = build_continuous_env(test)
    # VecNormalize(DummyVecEnv([lambda: build_continuous_env(dataset)]))

    eval_callback = CustomEvalCallback(env_eval, best_model_save_path=f'trained_models/{exp_name}/',
                                       n_eval_episodes=1,
                                       deterministic=True, render=True)
    log_callback = EnvTerminalStatsLoggingCallback()

    env_train = build_continuous_env(train)
    # SubprocVecEnv([lambda: build_continuous_env(train) for _ in range(num_train_envs)])

    learning_rate_schedule = lambda progress: custom_learning_rate_schedule(
        progress, max_lr=3e-4, min_lr=1e-5
    )

    action_dim = env_eval.action_space.shape[0]
    features_dim = 1024
    lstm_hidden = int(features_dim / 2)
    policy_kwargs = dict(
        features_extractor_class=GeGLUFFNNetExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim, num_blocks=8, dropout=0.5),
        net_arch=[round((lstm_hidden - action_dim) / 2)],
        share_features_extractor=False,
        normalize_images=False,
        # lstm_hidden_size=lstm_hidden,
        # n_lstm_layers=2,
        # lstm_kwargs=dict(dropout=0.3),
        activation_fn=th.nn.Identity,
        optimizer_class=th.optim.AdamW
    )

    batch = int(512 / num_train_envs)
    agent = SAC(
        policy='MultiInputPolicy',
        policy_kwargs=policy_kwargs,
        # n_epochs=1,
        # n_steps=batch,
        batch_size=batch,
        learning_rate=1e-5,
        env=env_train,
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
    train_agent(*load_dataset())
