import math
from typing import Callable

import pandas as pd
import torch
from pandas import DataFrame
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.callbacks import EnvTerminalStatsLoggingCallback
from src.rl.envs.continuous_env import PortfolioOptimizationEnv
from src.rl.envs.discrete_env import StockTradingEnv
from src.rl.loaders import split
from src.rl.models import PolicyGradient

time_window = 64


def load_dataset():
    dataset = pd.read_csv('C:/Users/oQaris/Desktop/Git/QFinance/data/pre/last_top_data_preprocess_norm_time.csv')
    return split(dataset, train_ratio=0.8, stratification=time_window)


def build_env(dataset: DataFrame, verbose=1):
    allux_cols = ['date', 'tic', 'lot', 'price']
    window_features = ['open',
                       'close',
                       'high',
                       'low',
                       'volume']
    time_features = ['vix'] + [f for f in dataset.columns if f.endswith('_sin') or f.endswith('_cos')]
    indicators = [f for f in dataset.columns if
                  f not in allux_cols and f not in window_features and f not in time_features][1:]
    env_kwargs = {
        'initial_amount': 1000000,
        'fee_ratio': 0.003,
        'time_window': time_window,
        'window_features': window_features,
        'time_features': time_features,
        'indicators_features': indicators,
        'reward_type': 'log',
        'verbose': verbose
    }
    dataset = dataset[dataset['tic'].isin(
        ['AFLT', 'GMKN', 'MOEX', 'TCSG', 'MAGN', 'LKOH', 'NLMK', 'OZON', 'POLY', 'SBER', 'VKCO', 'YDEX'])].copy()
    env = PortfolioOptimizationEnv(dataset, **env_kwargs)
    # check_env(env)
    return env


def build_discrete_env(dataset: DataFrame, verbose=1):
    # todo создавать автоматически
    ind_list = [
        'macd',
        'macdh',
        'rsi_14',
        'atr_14',
        'adx',
        'cci_14',
        'stochrsi',
        'wr_14',
        'pdi',
        'ndi',
        'trix',
        'dma',
        'cmo',
        'close_14_roc',
        'return_lag_1',
        'return_lag_7',
        'return_lag_29',
        'P/FCF',
        'P/E',
        'P/B',
        'P/S',
        'P/CF',
        'EV/S',
        'EV/EBITDA',
        'EV/EBIT',
        'Коэффициент долга',
        'D/E',
        'CAPEX/Выручка',
        'NetDebt/EBITDA',
        'Долг/EBITDA',
        'ROE',
        'ROA',
        'Return on Sales',
        'ROIC',
        'ROCE',
        'Net Margin',
        'Операционная маржа',
        'EBITDA рентабельность',
        'Тек. ливкидность'
    ]
    env_kwargs = {
        'initial_amount': 1000000,
        'comission_fee_pct': 0.003,
        'tech_indicator_list': ind_list,
        'verbose': verbose
    }
    return StockTradingEnv(df=dataset, **env_kwargs)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Создает линейный планировщик для изменения значения.
    :param initial_value: начальное значение, например, начальный learning rate.
    :return: функция, принимающая текущий прогресс (от 1 до 0) и возвращающая значение learning rate.
    """

    def schedule(progress_remaining: float) -> float:
        """
        Возвращает значение параметра в зависимости от оставшегося прогресса.
        :param progress_remaining: оставшийся прогресс обучения (от 1.0 до 0.0).
        :return: текущее значение параметра (learning rate).
        """
        return progress_remaining * initial_value

    return schedule


def cosine_annealing_schedule(progress_remaining):
    """
    Cosine Annealing Schedule для learning_rate.
    progress_remaining — значение от 1 (начало) до 0 (конец обучения).
    """
    initial_rate = 5e-4  # Начальная скорость обучения
    final_rate = 1e-4  # Минимальная скорость обучения
    cos_inner = (1 + math.cos(math.pi * (1 - progress_remaining))) / 2
    return final_rate + (initial_rate - final_rate) * cos_inner


def train_PG(env_train):
    agent = PolicyGradient(env_train,
                           policy_kwargs=dict(time_window=time_window,
                                              initial_features=len(env_train.window_features)))
    try:
        agent.train(episodes=500)
    except KeyboardInterrupt:
        print('Обучение прервано вручную. Сохраняем модель...')
    finally:
        model_path = 'trained_models/policy_EI3.pt'
        torch.save(agent.train_policy.state_dict(), model_path)
        print('Модель успешно сохранена.')


def train_agent(dataset):
    print('Инициализация среды...')
    # Separate evaluation env
    # eval_env = gym.make('Pendulum-v1')
    # Use deterministic actions for evaluation
    # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
    #                              log_path='./logs/', eval_freq=500,
    #                              deterministic=True, render=False)

    exp_name = 'MaskablePPO_discrete'
    num_envs = 8
    env_train = SubprocVecEnv([lambda: build_discrete_env(dataset) for _ in range(num_envs)])

    env_callback = EnvTerminalStatsLoggingCallback()

    # добавить приоритизированный буфер

    total_timesteps = 50_000_000
    agent = MaskablePPO(
        policy='MlpPolicy',
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
            callback=env_callback,
            progress_bar=True,
            tb_log_name=exp_name
        )
    except KeyboardInterrupt:
        print('Обучение прервано вручную. Сохраняем модель...')
    finally:
        agent.save(f'trained_models/{exp_name}')
        print('Модель успешно сохранена.')


if __name__ == '__main__':
    train, _ = load_dataset()
    # env_train = build_env(train, verbose=1)
    # env_train = build_discrete_env(train, verbose=1)
    train_agent(train)
