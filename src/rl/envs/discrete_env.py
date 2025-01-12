from __future__ import annotations

import numpy as np
import pandas as pd
import quantstats as qs
from gymnasium import spaces
from line_profiler import profile
from typing_extensions import override

from src.rl.algs import utils
from src.rl.algs.utils import plot_with_risk_free, calculate_periods_per_year
from src.rl.envs.base_env import BaseEnv


class StockTradingEnv(BaseEnv):

    def __init__(
            self,
            df: pd.DataFrame,
            initial_amount: int,
            tech_indicator_list: list[str],
            hmax: int = 100,
            comission_fee_pct: float = 0.003,
            valuation_column='price',
            time_column='date',
            tic_column='tic',
            lot_column='lot',
            verbose=0,
            time_index=0,
            previous_state=None,
    ):
        super().__init__(df)
        # данные
        self._df = self._df.sort_values(['date', 'tic'], ignore_index=True)
        self._df.index = self._df['date'].factorize()[0]

        self.num_periods = len(self._df.index.unique())
        self.time_index = time_index
        self.data = self._df.loc[self.time_index, :]

        # среда
        self.hmax = hmax
        self.stock_dim = len(self._df.tic.unique())
        self.num_stock_shares = [0] * self.stock_dim
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = self.sell_cost_pct = [comission_fee_pct] * self.stock_dim
        self.tech_indicator_list = tech_indicator_list
        self.verbose = verbose
        self.previous_state = previous_state

        # столбцы
        self.valuation_column = valuation_column
        self.time_column = time_column
        self.tic_column = tic_column
        self.lot_column = lot_column

        # для фреймворка Gym
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        state_space = 1 + 2 * self.stock_dim + len(tech_indicator_list) * self.stock_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32)
        self.metadata = {'render_modes': ['human'], 'render_fps': 1}
        self.render_mode = 'human'

        # инициализация состояния
        self.state = self._initiate_state()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # память эпизода
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1: 1 + self.stock_dim])
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.account_value_memory = [self.initial_amount]
        self.state_memory = []
        self.date_memory = [self._get_date()]

        # Оптимизация: предвычисляем лоты для каждого тикера
        lots_per_tic = self._df.groupby(self.tic_column)[self.lot_column].unique()
        if any(len(lots) > 1 for lots in lots_per_tic):
            raise ValueError("Лоты для одного тикера не совпадают во всем датасете.")
        self.lots = lots_per_tic.values.flatten()

    @profile
    def _sell_stock(self, index, action):
        if self.state[index + self.stock_dim + 1] <= 0:
            return 0

        lot = self.lots[index].item()
        sell_num_shares = min(-action * lot, self.state[index + self.stock_dim + 1].item())

        sell_amount = (
                self.state[index + 1].item()
                * sell_num_shares
                * (1 - self.sell_cost_pct[index])
        )
        self.state[0] += sell_amount

        if self.state[0] < 0:
            raise ValueError('Недостаточно средств')

        self.state[index + self.stock_dim + 1] -= sell_num_shares
        if self.state[index + self.stock_dim + 1] < 0:
            raise ValueError('Количество акций не может быть отрицательным')

        self.cost += (
                self.state[index + 1].item()
                * sell_num_shares
                * self.sell_cost_pct[index]
        )
        self.trades += 1

        return sell_num_shares

    @profile
    def _buy_stock(self, index, action):
        lot = self.lots[index].item()
        available_amount = int(
            self.state[0].item() / (self.state[index + 1].item() * lot * (1 + self.buy_cost_pct[index])))

        if available_amount <= 0:
            return 0

        buy_num_shares = min(available_amount, action * lot)
        buy_amount = (
                self.state[index + 1].item()
                * buy_num_shares
                * (1 + self.buy_cost_pct[index])
        )
        self.state[0] -= buy_amount

        if self.state[0] < 0:
            raise ValueError('Недостаточно средств')

        self.state[index + self.stock_dim + 1] += buy_num_shares
        if self.state[index + self.stock_dim + 1] < 0:
            raise ValueError('Количество акций не может быть отрицательным')

        self.cost += self.state[index + 1].item() * buy_num_shares * self.buy_cost_pct[index]
        self.trades += 1

        return buy_num_shares

    def get_terminal_stats(self):
        metrics_df = pd.DataFrame({
            'date': self.date_memory,
            'returns': self.account_value_memory,
            'portfolio_values': self.account_value_memory
        })
        metrics_df['returns'] = metrics_df['returns'].pct_change(1).dropna()
        # Убедимся, что дата является DatetimeIndex
        metrics_df['date'] = pd.to_datetime(metrics_df['date'])
        metrics_df.set_index('date', inplace=True)

        sharpe_ratio, sortino_ratio = utils.sharpe_sortino(metrics_df)
        max_draw_down = -qs.stats.max_drawdown(metrics_df['portfolio_values'])

        positions = np.array(self.state_memory)[:, self.stock_dim + 1:2 * self.stock_dim + 1]
        mean_position_tic = np.count_nonzero(positions, axis=1).mean()

        # todo учесть комиссию продажи
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        return {
            'profit': end_total_asset / self.initial_amount,
            'max_draw_down': max_draw_down,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'fee_ratio': self.cost / self.initial_amount,
            'mean_position_tic': mean_position_tic,
            'mean_transactions': self.trades / (self.num_periods - 1),
            'mean_reward': np.mean(self.rewards_memory),
            'std_reward': np.std(self.rewards_memory),
        }

    @profile
    def step(self, actions):
        terminal = self.is_terminal_state()
        info = {}

        actions = actions * self.hmax
        actions = actions.astype(int)  # This cast might be redundant, but good practice.
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)

        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])

        self.log_transactions(actions)
        self.actions_memory.append(actions)

        # Переход к следующему состоянию
        self.time_index += 1
        self.data = self._df.loc[self.time_index, :]
        self.state = self._update_state()

        # Запоминаем необходимые данные
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset.item())
        self.account_value_memory.append(end_total_asset.item())
        self.date_memory.append(self._get_date())

        # Вычисляем награду
        reward = end_total_asset - begin_total_asset
        reward = reward / self.initial_amount * 100
        self.rewards_memory.append(reward)
        self.state_memory.append(self.state)

        if terminal:
            terminal_stats = self.get_terminal_stats()
            info['terminal_stats'] = terminal_stats
            if self.verbose >= 1:
                print('\n=================================')
                print(f'day: {self.time_index}, episode: {self.episode}')
                print(f'begin_total_asset: {self.initial_amount}')
                for key, value in terminal_stats.items():
                    print(f'{key}: {value}')
                print('=================================')

        return self.state, reward, terminal, False, info

    def log_transactions(self, actions):
        if self.verbose > 1:
            date = self._get_date()
            tics = self.data[self.tic_column].values

            nonzero_indices = np.nonzero(actions)
            for idx in nonzero_indices[0]:
                elem = actions[idx]
                if elem > 0:
                    #todo ORUP lot
                    print(f'{date}: куплено {elem} акций {tics[idx]}')
                elif elem < 0:
                    print(f'{date}: продано {-elem} акций {tics[idx]}')

            if np.any(actions != 0):
                open_positions = np.count_nonzero(self.state[self.stock_dim + 1: 2 * self.stock_dim + 1])
                print(f'Открыто {open_positions} позиций:')
                held_shares = self.state[self.stock_dim + 1: 2 * self.stock_dim + 1]
                for idx in range(self.stock_dim):
                    if held_shares[idx] > 0:
                        print(f'{tics[idx]}={held_shares[idx]}', end=', ')
                print(f'\nСвободных денег: {self.state[0]}')

    @profile
    def action_masks(self):
        mask = []
        state = np.array(self.state)
        buy_cost_pct = np.array(self.buy_cost_pct)

        lot_values = self.data[self.lot_column].values[:self.stock_dim]
        prices = state[1:self.stock_dim + 1]
        max_buy = (state[0] / (prices * lot_values * (1 + buy_cost_pct))).astype(int)
        max_sell = state[self.stock_dim + 1:2 * self.stock_dim + 1].astype(int)

        lot_counts = np.arange(-self.hmax, self.hmax + 1)
        for i in range(self.stock_dim):
            action_mask = np.zeros(2 * self.hmax + 1, dtype=int)

            # Векторизированное создание маски
            buy_mask = (lot_counts > 0) & (lot_counts <= max_buy[i])
            sell_mask = (lot_counts < 0) & (-lot_counts <= max_sell[i])
            hold_mask = (lot_counts == 0)

            action_mask[buy_mask | sell_mask | hold_mask] = 1
            mask.append(action_mask)

        return np.concatenate(mask)

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ):
        super().reset(seed=seed)
        # initiate state
        self.time_index = 0
        self.data = self._df.loc[self.time_index, :]
        self.state = self._initiate_state()

        if self.previous_state is None:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1: 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1: 1 + self.stock_dim])
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.account_value_memory = [self.initial_amount]
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        return self.state, {}

    def _initiate_state(self):
        if self.previous_state is None:
            # For Initial State
            state = (
                    [self.initial_amount]
                    + self.data[self.valuation_column].values.tolist()
                    + self.num_stock_shares
                    + sum(
                (
                    self.data[tech].values.tolist()
                    for tech in self.tech_indicator_list
                ),
                [],
            )
            )  # append initial stocks_share to initial state, instead of all zero
        else:
            # Using Previous State
            state = (
                    [self.previous_state[0]]
                    + self.data[self.valuation_column].values.tolist()
                    + self.previous_state[
                      (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                      ]
                    + sum(
                (
                    self.data[tech].values.tolist()
                    for tech in self.tech_indicator_list
                ),
                [],
            )
            )
        return np.array(state, dtype=np.float32)

    @profile
    def _update_state(self):
        state = (
                [self.state[0]]
                + self.data[self.valuation_column].values.tolist()
                + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                + sum(
            (
                self.data[tech].values.tolist()
                for tech in self.tech_indicator_list
            ),
            [],
        )
        )
        return np.array(state, dtype=np.float32)

    def _get_date(self):
        return self.data[self.time_column].unique()[0]

    def render(self):
        if self.is_terminal_state():
            plot_with_risk_free(self.account_value_memory, calculate_periods_per_year(self._df))

    @override
    def is_terminal_state(self):
        return self.time_index >= self.num_periods - 2

    @override
    def get_portfolio_size_history(self):
        return self.account_value_memory
