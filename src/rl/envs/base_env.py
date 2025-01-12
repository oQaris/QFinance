from abc import abstractmethod

import gymnasium as gym
import pandas as pd


class BaseEnv(gym.Env):

    def __init__(self, df: pd.DataFrame, kwargs=None):
        self._df = df

    @abstractmethod
    def is_terminal_state(self) -> bool:
        pass

    @abstractmethod
    def get_portfolio_size_history(self) -> list[float]:
        pass
