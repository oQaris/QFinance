from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
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
        self.terminal = False
        self.verbose = verbose
        self.previous_state = previous_state
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1: 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]

    def _sell_stock(self, index, action):
        # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
        # Sell only if the price is > 0 (no missing data in this particular date)
        # perform sell action based on the sign of the action
        if self.state[index + self.stock_dim + 1] <= 0:
            return 0

        # Sell only if current asset is > 0
        sell_num_shares = min(
            abs(action), self.state[index + self.stock_dim + 1]
        ) * self.data[self.lot_column].values[index]
        sell_amount = (
                self.state[index + 1]
                * sell_num_shares
                * (1 - self.sell_cost_pct[index])
        )
        # update balance
        self.state[0] += sell_amount

        self.state[index + self.stock_dim + 1] -= sell_num_shares
        self.cost += (
                self.state[index + 1]
                * sell_num_shares
                * self.sell_cost_pct[index]
        )
        self.trades += 1

        return sell_num_shares

    def _buy_stock(self, index, action):
        # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
        available_amount = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))

        # update balance
        buy_num_shares = min(available_amount, action) * self.data[self.lot_column].values[index]
        buy_amount = (
                self.state[index + 1]
                * buy_num_shares
                * (1 + self.buy_cost_pct[index])
        )
        self.state[0] -= buy_amount

        self.state[index + self.stock_dim + 1] += buy_num_shares
        self.cost += (
                self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
        )
        self.trades += 1

        return buy_num_shares

    def step(self, actions):
        self.terminal = self.time_index >= len(self.df.index.unique()) - 2

        if self.terminal and self.verbose >= 1:
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                    self.state[0]
                    + sum(np.array(self.state[1: (self.stock_dim + 1)])
                          * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]))
                    - self.asset_memory[0])  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                        (252 ** 0.5)
                        * df_total_value["daily_return"].mean()
                        / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]

            print(f"day: {self.time_index}, episode: {self.episode}")
            print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            print(f"end_total_asset: {end_total_asset:0.2f}")
            print(f"total_reward: {tot_reward:0.2f}")
            print(f"total_cost: {self.cost:0.2f}")
            print(f"total_trades: {self.trades}")
            if df_total_value["daily_return"].std() != 0:
                print(f"Sharpe: {sharpe:0.3f}")
            print("=================================")

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

        # state: s -> s+1
        self.time_index += 1
        self.data = self.df.loc[self.time_index, :]
        self.state = self._update_state()

        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())
        self.reward = end_total_asset - begin_total_asset
        self.rewards_memory.append(self.reward)
        self.reward = self.reward / self.initial_amount * 100
        self.state_memory.append(self.state)  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}

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
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
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
        df_date.columns = ["date"]

        state_list = self.state_memory
        df_states = pd.DataFrame(
            state_list,
            columns=[
                "cash",
                "Bitcoin_price",
                "Gold_price",
                "Bitcoin_num",
                "Gold_num",
                "Bitcoin_Disable",
                "Gold_Disable",
            ],
        )
        df_states.index = df_date.date
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data[self.tic_column].values
        df_actions.index = df_date.date
        return df_actions
