from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pandas as pd
import quantstats as qs
from gymnasium import spaces

from src.rl.algs.distributor import distribute_optimally


def _softmax_normalization(actions):
    """Normalizes the action vector using softmax function.

    Returns:
        Normalized action vector (portfolio vector).
    """
    numerator = np.exp(actions)
    denominator = np.sum(np.exp(actions))
    softmax_output = numerator / denominator
    return softmax_output


def _custom_softmax(x):
    # Маска для ненулевых элементов
    mask = x != 0
    # Применяем экспоненту к ненулевым элементам
    exp_values = np.exp(x[mask])
    # Нормализуем ненулевые элементы
    exp_values /= np.sum(exp_values)
    # Заменяем ненулевые элементы на их нормализованные значения
    result = np.zeros_like(x)
    result[mask] = exp_values
    return result


def _sum_normalization(actions):
    return actions / np.sum(actions)


class PortfolioOptimizationEnv(gym.Env):

    def __init__(
            self,
            df,
            initial_amount,
            normalize_df='by_previous_time',
            reward_scaling=1,
            comission_fee_model='trf',
            comission_fee_pct=0,
            features=['close', 'high', 'low'],
            valuation_feature='close',
            time_column='date',
            tic_column='tic',
            tics_in_portfolio='all',
            time_window=50,
            verbose=False,
    ):
        self.commission_paid = 0
        self.num_of_transactions = 0
        self._reward = 0
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._verbose = verbose
        # initialize price variation
        self._df_price_variation = None
        # preprocess data
        self._preprocess_data(normalize_df, tics_in_portfolio)
        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = (
            len(self._tic_list)
            if tics_in_portfolio == 'all'
            else len(tics_in_portfolio)
        )
        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        self._portfolio_value = self._initial_amount
        self._terminal = False
        # Необходимо для OpenAI Gym
        action_space = 1 + self.portfolio_size
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._features), len(self._tic_list), self._time_window),
        )
        self._reset_memory()
        self.tic_indices_cache = {
            tic: self._df[self._tic_column] == tic
            for tic in self._tic_list
        }

    def step(self, actions):
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            metrics_df = pd.DataFrame(
                {
                    'date': self._date_memory,
                    'returns': self._portfolio_return_memory,
                    'portfolio_values': self._asset_memory['final'],
                }
            )
            metrics_df.set_index('date', inplace=True)
            print('=================================')
            print('Initial portfolio value:{}'.format(self._asset_memory['final'][0]))
            print(f'Final portfolio value: {self._portfolio_value}')
            print(
                'Final accumulative portfolio value: {}'.format(
                    self._portfolio_value / self._asset_memory['final'][0]
                )
            )
            print(
                'Maximum DrawDown: {}'.format(
                    qs.stats.max_drawdown(metrics_df['portfolio_values'])
                )
            )
            print('Sharpe ratio: {}'.format(qs.stats.sharpe(metrics_df['returns'])))
            print('Commission paid: {}'.format(self.commission_paid))
            print('Number of transactions: {}'.format(self.num_of_transactions))
            print('=================================')
            return self._state, self._reward, self._terminal, False, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = _custom_softmax(actions)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            current_portfolio, comission = self._rebalance_portfolio(weights)  # weights*self._portfolio_value, 1000 #
            self.commission_paid += comission
            self._portfolio_value = current_portfolio.sum()

            # save initial portfolio value of this time step
            self._asset_memory['initial'].append(self._portfolio_value)

            # load next state
            self._time_index += 1
            # calc _price_variation
            self._state, self._info = self._get_state_and_info_from_time_index(self._time_index)
            self._info['trf_mu'] = 1 - (comission / self._portfolio_value)

            # time passes and time variation changes the portfolio distribution
            new_portfolio = current_portfolio * self._price_variation

            # calculate new portfolio value and weights
            self._portfolio_value = np.sum(new_portfolio)
            final_weights = new_portfolio / self._portfolio_value
            self._info['real_weights'] = final_weights

            # save final portfolio value and weights of this time step
            self._asset_memory['final'].append(self._portfolio_value)
            self._final_weights.append(final_weights)

            # save date memory
            self._date_memory.append(self._info['end_time'])

            # define portfolio return
            rate_of_return = self._asset_memory['final'][-1] / self._asset_memory['final'][-2]
            portfolio_return = rate_of_return - 1

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)

            # Define portfolio return
            self._reward = portfolio_return
            self._reward = self._reward * self._reward_scaling

        return self._state, self._reward, self._terminal, False, self._info

    def eval_trades(self, rest_cash):
        tic_counts = self._tic_counts_memory[-1]
        diff_tic_counts = tic_counts - self._tic_counts_memory[-2]
        date = self._sorted_times[self._time_index]
        nonzero_indices = np.nonzero(diff_tic_counts)[0]
        tics = self._df['tic'].unique()
        for idx in nonzero_indices:
            elem = diff_tic_counts[idx]
            self.num_of_transactions += 1
            if self._verbose:
                if elem > 0:
                    print(f'{date}: куплено {elem} акций {tics[idx]}')
                elif elem < 0:
                    print(f'{date}: продано {-elem} акций {tics[idx]}')
        if self._verbose:
            print(f'Открыто {len(np.nonzero(tic_counts)[0])} позиций:')
            for idx in np.nonzero(tic_counts)[0]:
                elem = tic_counts[idx]
                print(f'{tics[idx]}={elem}', end=', ')
            print(f'\nСвободных денег: {rest_cash}')

    def _rebalance_portfolio(self, weights):
        """
        Рассчитывает количество акций для ребалансировки портфеля.

        :param weights: одномерный массив NumPy, где 0-й элемент — это доля денежных средств,
                        а остальные элементы — это доля по каждой акции.

        :return: Новая распределённая стоимость портфеля + уже учтённая в нём комиссия
        """
        data_slice = self._df[
            (self._df[self._time_column] == self._sorted_times[self._time_index])
        ].set_index('tic')

        # В начале - денежные средства
        price = np.array([1.0] + data_slice['close_orig'].tolist())
        lot_size = np.array([1.0] + data_slice['lot'].tolist())
        multipliers = price * lot_size

        comission = 0
        counter = 0
        while True:
            counter += 1
            if counter > 3:
                print('что то не так')
            real_values, comission, tic_counts = self.calc_fee_portfolio(price, weights, multipliers, comission)
            rest_cash = real_values[0] - comission
            if rest_cash >= 0:
                break

        self._tic_counts_memory.append(tic_counts)
        self.eval_trades(rest_cash)

        real_values[0] = rest_cash
        return real_values, comission

    def calc_fee_portfolio(self, price, weights, multipliers, reserved):
        to_distribute = self._portfolio_value - reserved
        real_values = distribute_optimally(weights, multipliers, to_distribute)
        if real_values.sum() > to_distribute:
            raise ValueError('error distribute_optimally')
        real_values[0] += (reserved + to_distribute - real_values.sum())  # Нераспределённый остаток

        # Считаем количество тикеров акций (без учёта кеша)
        tic_counts = np.array([int(np.round(x)) for x in (real_values[1:] / price[1:])])
        diff_tic_counts = tic_counts - self._tic_counts_memory[-1]

        # Расчёт комиссии за сделки
        turnover = (np.abs(diff_tic_counts) * price[1:]).sum()
        comission = turnover * self._comission_fee_pct

        return real_values, comission, tic_counts

    def reset(self, seed=None, options=None):
        self.commission_paid = 0
        self.num_of_transactions = 0
        self._reward = 0
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state, self._info = self._get_state_and_info_from_time_index(
            self._time_index
        )
        self._portfolio_value = self._initial_amount
        self._terminal = False

        return self._state, self._info

    def _get_state_and_info_from_time_index(self, time_index):
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

        # define data to be used in this time step
        self._data = self._df[
            (self._df[self._time_column] >= start_time)
            & (self._df[self._time_column] <= end_time)
            ][[self._time_column, self._tic_column] + self._features + ['lot']]

        # define price variation of this time_step
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self._time_column] == end_time
            ][self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        # state = None
        # tic_indices = {tic: self._data[self._tic_column] == tic for tic in self._tic_list}
        # for tic in self._tic_list:
        #     tic_data = self._data.loc[tic_indices[tic], self._features].to_numpy().T
        #     tic_data = tic_data[..., np.newaxis]
        #     state = tic_data if state is None else np.append(state, tic_data, axis=2)

        # state = np.empty((len(self._features), self._time_window, len(self._tic_list)))
        # for i, tic in enumerate(self._tic_list):
        #     tic_data = self._data.loc[self.tic_indices_cache[tic], self._features].to_numpy().T
        #     tic_data = np.squeeze(tic_data)  # Убираем ось с длиной 1
        #     state[:, :, i] = tic_data

        # Выбираем все данные для тикеров за один вызов
        all_tic_data = self._data.set_index(self._tic_column).loc[self._tic_list, self._features].to_numpy()
        # Меняем размерность массива для работы с временем и тикерами
        all_tic_data = all_tic_data.reshape(len(self._tic_list), len(self._features), self._time_window)
        # Присваиваем данные в state
        state = all_tic_data.transpose(1, 2, 0)  # Меняем оси, чтобы привести к нужному формату

        # Первая ось - фичи, вторая - тикеры, третья - время
        state = state.transpose((0, 2, 1))
        info = {
            'tics': self._tic_list,
            'start_time': start_time,
            'start_time_index': time_index - (self._time_window - 1),
            'end_time': end_time,
            'end_time_index': time_index,
            'data': self._data,
            'price_variation': self._price_variation,
        }
        return state, info

    def _preprocess_data(self, normalize, tics_in_portfolio):
        """Orders and normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are 'by_previous_time', 'by_fist_time_window_value',
                'by_COLUMN_NAME' (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            tics_in_portfolio: List of ticker symbols to be considered as part of the
                portfolio. If 'all', all tickers of input data are considered.
        """
        # order time dataframe by tic and time
        self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df()
        # select only stocks in portfolio
        if tics_in_portfolio != 'all':
            self._df_price_variation = self._df_price_variation[
                self._df_price_variation[self._tic_column].isin(tics_in_portfolio)
            ]
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(
            self._df_price_variation[self._time_column]
        )
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype('float32')
        self._df_price_variation[self._features] = self._df_price_variation[
            self._features
        ].astype('float32')

    def _reset_memory(self):
        """Resets the environment's memory."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            'initial': [self._initial_amount],
            'final': [self._initial_amount],
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        self._tic_counts_memory = [
            np.array([0] * self.portfolio_size, dtype=np.uint32)
        ]
        # memorize datetimes
        self._date_memory = [date_time]

    def _normalize_dataframe(self, normalize):
        """ 'Normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are 'by_previous_time', 'by_fist_time_window_value',
                'by_COLUMN_NAME' (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.

        Note:
            If a custom function is used in the normalization, it must have an
            argument representing the environment's dataframe.
        """
        if type(normalize) == str:
            if normalize == 'by_fist_time_window_value':
                print(
                    'Normalizing {} by first time window value...'.format(
                        self._features
                    )
                )
                self._df = self._temporal_variation_df(self._time_window - 1)
            elif normalize == 'by_previous_time':
                print(f'Normalizing {self._features} by previous time...')
                self._df = self._temporal_variation_df()
            elif normalize.startswith('by_'):
                normalizer_column = normalize[3:]
                print(f'Normalizing {self._features} by {normalizer_column}')
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
        elif callable(normalize):
            print('Applying custom normalization function...')
            self._df = normalize(self._df)
        else:
            print('No normalization was performed.')

    def _temporal_variation_df(self, periods=1):
        """Calculates the temporal variation dataframe. For each feature, this
        dataframe contains the rate of the current feature's value and the last
        feature's value given a period. It's used to normalize the dataframe.

        Args:
            periods: Periods (in time indexes) to calculate temporal variation.

        Returns:
            Temporal variation dataframe.
        """
        df_temporal_variation = self._df.copy()
        prev_columns = []
        for column in self._features:
            prev_column = f'prev_{column}'
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(
                self._tic_column
            )[column].shift(periods=periods)
            df_temporal_variation[column] = (
                    df_temporal_variation[column] / df_temporal_variation[prev_column]
            )
        df_temporal_variation = (
            df_temporal_variation
            .drop(columns=prev_columns)
            .replace([float('inf'), -float('inf')], 1)
            .fillna(1)
            .reset_index(drop=True)
        )
        return df_temporal_variation

    def get_portfolio_size_history(self):
        return self._asset_memory['final']
