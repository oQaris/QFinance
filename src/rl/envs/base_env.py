from abc import abstractmethod

import gymnasium as gym


class BaseEnv(gym.Env):
    @abstractmethod
    def is_terminal_state(self) -> bool:
        pass

    @abstractmethod
    def get_portfolio_size_history(self) -> list[float]:
        pass
