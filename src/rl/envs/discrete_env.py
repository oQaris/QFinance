from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
import quantstats as qs
from gymnasium import spaces


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

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
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

    def _sell_stock(self, index, action):
        if self.state[index + self.stock_dim + 1] <= 0:
            return 0

        #todo не умножать на лот, а брать ближайшее кратное лоту, но только после перехода на динамический hmax
        lot = self.data[self.lot_column].values[index]
        sell_num_shares = min(-action * lot, self.state[index + self.stock_dim + 1])

        sell_amount = (
                self.state[index + 1]
                * sell_num_shares
                * (1 - self.sell_cost_pct[index])
        )
        self.state[0] += sell_amount

        if self.state[0] < 0:
            raise ValueError('asdfasdf')

        self.state[index + self.stock_dim + 1] -= sell_num_shares
        if self.state[index + self.stock_dim + 1] < 0:
            raise ValueError('asdfasdf3')

        self.cost += (
                self.state[index + 1]
                * sell_num_shares
                * self.sell_cost_pct[index]
        )
        self.trades += 1

        return sell_num_shares

    def _buy_stock(self, index, action):
        # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
        lot = self.data[self.lot_column].values[index]
        available_amount = int(self.state[0] / (self.state[index + 1] * lot * (1 + self.buy_cost_pct[index])))

        if available_amount <= 0:
            return 0

        # update balance
        buy_num_shares = min(available_amount, action * lot)
        buy_amount = (
                self.state[index + 1]
                * buy_num_shares
                * (1 + self.buy_cost_pct[index])
        )
        self.state[0] -= buy_amount

        if self.state[0] < 0:
            raise ValueError('asdfasdf')

        self.state[index + self.stock_dim + 1] += buy_num_shares
        if self.state[index + self.stock_dim + 1] < 0:
            raise ValueError('asdfasdf3')

        self.cost += (
                self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
        )
        self.trades += 1

        return buy_num_shares

    def get_terminal_stats(self):
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        df_total_value = pd.DataFrame(self.account_value_memory, columns=['account_value'])
        df_total_value['date'] = self.date_memory
        df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)

        sharpe_ratio = qs.stats.sharpe(df_total_value['daily_return'].dropna(), annualize=True)
        max_draw_down = qs.stats.max_drawdown(df_total_value['account_value'])

        return {
            'profit': end_total_asset / self.initial_amount,
            'max_draw_down': max_draw_down,
            'sharpe_ratio': sharpe_ratio,
            'fee_ratio': self.cost / self.initial_amount,
            'mean_position_tic': 0,  # todo
            'mean_transactions': self.trades / (self.num_periods - 1),
        }

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

        actions = actions * self.hmax  # actions initially is scaled between 0 to 1
        actions = actions.astype(int)  # convert into integer because we can't by fraction of shares
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

        return self.state, reward, terminal, False, {}

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
