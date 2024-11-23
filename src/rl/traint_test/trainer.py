import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.callbacks import EnvTerminalStatsLoggingCallback
from src.rl.envs.continuous_env import PortfolioOptimizationEnv
from src.rl.envs.discrete_env import StockTradingEnv
from src.rl.loaders import split
from src.rl.models import PolicyGradient

time_window = 1


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


def custom_learning_rate_schedule(progress: float, max_lr: float, min_lr: float) -> float:
    """
    Кастомное расписание для лёрнинг рейта.

    :param progress: Текущий прогресс обучения (от 0 до 1).
    :param max_lr: Максимальный лёрнинг рейт.
    :param min_lr: Минимальный лёрнинг рейт.
    :return: Значение лёрнинг рейта для текущего прогресса.
    """
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

    exp_name = 'PPO_lstm'
    num_envs = 16
    env_train = SubprocVecEnv([lambda: build_env(dataset) for _ in range(num_envs)])

    env_callback = EnvTerminalStatsLoggingCallback()

    learning_rate_schedule = lambda progress: custom_learning_rate_schedule(
        progress, max_lr=3e-4 * 2, min_lr=3e-4 / 3
    )

    total_timesteps = 50_000_000
    agent = RecurrentPPO(
        # policy=CustomDirichletRecurrentPolicy,
        policy="MultiInputLstmPolicy",
        learning_rate=learning_rate_schedule,
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
