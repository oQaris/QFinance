import warnings
from typing import Callable

import pandas as pd
from gymnasium.utils.env_checker import check_env

from src.rl.envs.base_env import BaseEnv
from src.rl.envs.continuous_env import PortfolioOptimizationEnv
from src.rl.envs.discrete_env import StockTradingEnv
from src.rl.loaders import split

warnings.filterwarnings("ignore",
                        message=".*For Box action spaces, we recommend using a symmetric and normalized space.*")
warnings.filterwarnings("ignore",
                        message=".*A Box observation space has an unconventional shape (neither an image, nor a 1D vector).*")
warnings.filterwarnings("ignore",
                        message=".*It seems a Box observation space is an image but the lower and upper bounds are not [0, 255].*")
warnings.filterwarnings("ignore",
                        message=".*A Box observation space minimum value is -infinity. This is probably too low.*")
warnings.filterwarnings("ignore",
                        message=".*A Box observation space maximum value is infinity. This is probably too high.*")
warnings.filterwarnings("ignore",
                        message=".*Not able to test alternative render modes due to the environment not having a spec.*")
warnings.filterwarnings("ignore",
                        message=".*Evaluation environment is not wrapped with a.*")

time_window = 1
initial_amount = 500_000
fee_ratio = 0.003
subset_tics = None
# subset_tics = ['AFLT', 'GMKN', 'MOEX', 'TCSG', 'MAGN', 'LKOH', 'NLMK', 'OZON', 'POLY', 'SBER', 'VKCO', 'YDEX']


def load_datasets():
    # todo 2024-01-04_2024-10-04_5_MIN_final
    dataset = pd.read_csv('../../../data/pre/2024-06-01_2025-01-01_DAY_final.csv')
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


EnvBuildType = Callable[[pd.DataFrame, bool, int], BaseEnv]


def build_continuous_env(dataset: pd.DataFrame, env_check=False, verbose=1) -> BaseEnv:
    window_features, time_features, indicators = prepare_columns(dataset.columns)
    return _build_env(PortfolioOptimizationEnv,
                      dataset,
                      env_check,
                      initial_amount=initial_amount,
                      fee_ratio=fee_ratio,
                      time_window=time_window,
                      window_features=window_features,
                      time_features=time_features,
                      indicators_features=indicators,
                      verbose=verbose)


def build_discrete_env(dataset: pd.DataFrame, env_check=False, verbose=1) -> BaseEnv:
    _, _, indicators = prepare_columns(dataset.columns)
    return _build_env(StockTradingEnv,
                      dataset,
                      env_check,
                      initial_amount=initial_amount,
                      comission_fee_pct=fee_ratio,
                      tech_indicator_list=indicators,
                      verbose=verbose)


def _build_env(env_class: type[BaseEnv], dataset, env_check, **env_kwargs) -> BaseEnv:
    if subset_tics is not None:
        dataset = dataset[dataset['tic'].isin(subset_tics)]
    env = env_class(dataset.copy(), **env_kwargs)
    if env_check:
        check_env(env)
    return env
