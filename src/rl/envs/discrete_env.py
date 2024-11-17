from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
import quantstats as qs
from gymnasium import spaces
from line_profiler import profile


class StockTradingEnv(gym.Env):

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
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df['date'].factorize()[0]
        self.df = df
        self.num_periods = len(self.df.index.unique())

        self.time_index = time_index
        self.hmax = hmax
        self.stock_dim = len(df.tic.unique())
        self.num_stock_shares = [0] * self.stock_dim
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = self.sell_cost_pct = [comission_fee_pct] * self.stock_dim
        self.tech_indicator_list = tech_indicator_list

        self.action_space = spaces.MultiDiscrete([2 * self.hmax + 1] * self.stock_dim)  # Buy/Sell/Hold (-hmax to +hmax)
        state_space = 1 + 2 * self.stock_dim + len(tech_indicator_list) * self.stock_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_space,))

        self.valuation_column = valuation_column
        self.time_column = time_column
        self.tic_column = tic_column
        self.lot_column = lot_column

        self.data = self.df.loc[self.time_index, :]
        self.verbose = verbose
        self.previous_state = previous_state
        # initalize state
        self.state = self._initiate_state()

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
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
        # we need sometimes to preserve the state in the middle of trading process
        self.state_memory = []
        self.date_memory = [self._get_date()]

    @profile
    def _sell_stock(self, index, action):
        if self.state[index + self.stock_dim + 1] <= 0:
            return 0

        sell_num_shares = min(-action, self.state[index + self.stock_dim + 1])

        sell_amount = (
                self.state[index + 1]
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
                self.state[index + 1]
                * sell_num_shares
                * self.sell_cost_pct[index]
        )
        self.trades += 1

        return sell_num_shares

    @profile
    def _buy_stock(self, index, action):
        lot = self.data[self.lot_column].values[index]  # todo оптимизировать
        available_amount = int(self.state[0] / (self.state[index + 1] * lot * (1 + self.buy_cost_pct[index])))

        if available_amount <= 0:
            return 0

        # update balance
        buy_num_shares = min(available_amount, action)
        buy_amount = (
                self.state[index + 1]
                * buy_num_shares
                * (1 + self.buy_cost_pct[index])
        )
        self.state[0] -= buy_amount

        if self.state[0] < 0:
            raise ValueError('Недостаточно средств')

        self.state[index + self.stock_dim + 1] += buy_num_shares
        if self.state[index + self.stock_dim + 1] < 0:
            raise ValueError('Количество акций не может быть отрицательным')

        self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
        self.trades += 1
        # todo добавить лог

        return buy_num_shares

    def get_terminal_stats(self):
        # todo учесть комиссию продажи
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        df_total_value = pd.DataFrame(self.account_value_memory, columns=['account_value'])
        df_total_value['date'] = self.date_memory
        df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)

        sharpe_ratio = qs.stats.sharpe(df_total_value['daily_return'].dropna())
        sortino_ratio = qs.stats.sortino(df_total_value['daily_return'].dropna())
        max_draw_down = qs.stats.max_drawdown(df_total_value['account_value'])

        return {
            'profit': end_total_asset / self.initial_amount,
            'max_draw_down': max_draw_down,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'fee_ratio': self.cost / self.initial_amount,
            'mean_position_tic': 0,  # todo
            'mean_transactions': self.trades / (self.num_periods - 1),
        }

    @profile
    def step(self, actions):
        terminal = self.time_index >= self.num_periods - 2
        info = {}

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

        actions = actions - self.hmax
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

        self.actions_memory.append(actions)

        # Переход к следующему состоянию
        self.time_index += 1
        self.data = self.df.loc[self.time_index, :]
        self.state = self._update_state()

        # Запоминаем необходимые данные
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)
        self.account_value_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        # Вычисляем награду
        reward = end_total_asset - begin_total_asset
        self.rewards_memory.append(reward)
        reward = reward / self.initial_amount * 100
        self.state_memory.append(self.state)  # add current state in state_recorder for each step

        return self.state, reward, terminal, False, info

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
        # initiate state
        self.time_index = 0
        self.data = self.df.loc[self.time_index, :]
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
        return state

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
        return state

    def _get_date(self):
        return self.data[self.time_column].unique()[0]

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        state_list = self.state_memory
        df_states = pd.DataFrame(
            state_list,
            columns=[
                'cash',
                'Bitcoin_price',
                'Gold_price',
                'Bitcoin_num',
                'Gold_num',
                'Bitcoin_Disable',
                'Gold_Disable',
            ],
        )
        df_states.index = df_date.date
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {'date': date_list, 'account_value': asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data[self.tic_column].values
        df_actions.index = df_date.date
        return df_actions

    def get_portfolio_size_history(self):
        return self.account_value_memory
