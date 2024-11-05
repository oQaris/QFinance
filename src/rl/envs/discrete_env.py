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
            reward_scaling: float = 1e-4,
            print_verbosity=1,
            day=0,
            initial=True,
            previous_state=[],
            model_name="",
            mode="",
            iteration="",
    ):
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df['date'].factorize()[0]
        self.df = df

        self.day = day
        self.hmax = hmax
        self.stock_dim = len(df.tic.unique())
        self.num_stock_shares = [0] * self.stock_dim
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = self.sell_cost_pct = [comission_fee_pct] * self.stock_dim
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        state_space = 1 + 2 * self.stock_dim + len(tech_indicator_list) * self.stock_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_space,))

        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
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
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()

    def _sell_stock(self, index, action):
        if self.state[index + 2 * self.stock_dim + 1] != True:
            # check if the stock is able to sell, for simlicity we just add it in techical index
            # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
            # Sell only if the price is > 0 (no missing data in this particular date)
            # perform sell action based on the sign of the action
            if self.state[index + self.stock_dim + 1] > 0:
                # Sell only if current asset is > 0
                sell_num_shares = min(
                    abs(action), self.state[index + self.stock_dim + 1]
                )
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
            else:
                sell_num_shares = 0
        else:
            sell_num_shares = 0

        return sell_num_shares

    def _buy_stock(self, index, action):
        if self.state[index + 2 * self.stock_dim + 1] != True:
            # check if the stock is able to buy
            # if self.state[index + 1] >0:
            # Buy only if the price is > 0 (no missing data in this particular date)
            available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
            )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
            # print('available_amount:{}'.format(available_amount))

            # update balance
            buy_num_shares = min(available_amount, action)
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
        else:
            buy_num_shares = 0
        return buy_num_shares

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
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
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(int)  # convert into integer because we can't by fraction of shares
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
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
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
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
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.initial_amount]
                        + self.data.close.values.tolist()
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
                # for single stock
                state = (
                        [self.initial_amount]
                        + [self.data.close]
                        + [0] * self.stock_dim
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.previous_state[0]]
                        + self.data.close.values.tolist()
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
            else:
                # for single stock
                state = (
                        [self.previous_state[0]]
                        + [self.data.close]
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                    [self.state[0]]
                    + self.data.close.values.tolist()
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum(
                (
                    self.data[tech].values.tolist()
                    for tech in self.tech_indicator_list
                ),
                [],
            )
            )

        else:
            # for single stock
            state = (
                    [self.state[0]]
                    + [self.data.close]
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
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
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions
