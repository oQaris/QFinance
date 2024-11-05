from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pandas as pd
import quantstats as qs
from gymnasium import spaces

from src.rl.algs.distributor import discrete_allocation_custom, minimize_transactions


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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


class PortfolioOptimizationEnv(gym.Env):

    def __init__(
            self,
            df,
            initial_amount,
            window_features=['close', 'high', 'low', 'volume'],
            indicators_features=['macd', 'rsi_14', 'adx', 'return_lag_7', 'P/E', 'EV/EBITDA'],
            time_features=['vix', 'turbulence'],
            valuation_column='price',
            time_column='date',
            tic_column='tic',
            lot_column='lot',
            time_window=64,
            reward_scaling=10,
            comission_fee_pct=0.003,
            verbose=0,
    ):
        self._state = None
        self._info = None
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
        self._window_features = window_features
        self._indicators_features = indicators_features
        self._time_features = time_features
        self._valuation_column = valuation_column
        self._lot_column = lot_column

        self._verbose = verbose
        # initialize price variation
        self._df_price_variation = None
        # preprocess data
        self._preprocess_data()
        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = len(self._tic_list)
        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        self._portfolio_value = self._initial_amount
        self._terminal = False

        # Необходимо для OpenAI Gym
        self.action_space = spaces.Box(low=0, high=1, shape=(self.portfolio_size + 1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'price_data': spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self._window_features), self.portfolio_size, self._time_window),
                dtype=np.float32
            ),
            'indicators': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.portfolio_size, len(self._indicators_features)), dtype=np.float32
            ),
            'common_data': spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self._time_features),), dtype=np.float32
            ),
            'portfolio_dist': spaces.Box(
                low=0, high=1, shape=(self.portfolio_size + 1,), dtype=np.float32
            )
        })
        self._reset_memory()

    def is_pred_terminal_state(self):
        return self._time_index >= len(self._sorted_times) - 3

    def get_terminal_stats(self):
        metrics_df = pd.DataFrame({
            'date': self._date_memory,
            'returns': self._portfolio_return_memory,
            'portfolio_values': self._asset_memory['final'],
            'tic_counts': self._tic_counts_memory
        })
        metrics_df.set_index('date', inplace=True)

        profit = self._portfolio_value / self._initial_amount
        max_draw_down = qs.stats.max_drawdown(metrics_df['portfolio_values'])
        sharpe_ratio = qs.stats.sharpe(metrics_df['returns'])
        fee_ratio = self.commission_paid / self._initial_amount

        positions = np.stack(metrics_df['tic_counts'].values[1:])  # Первый элемент не учитываем, т.к. там нули
        mean_position_tic = np.count_nonzero(positions, axis=1).mean()
        mean_transactions = self.num_of_transactions / metrics_df.shape[0]

        return profit, max_draw_down, sharpe_ratio, fee_ratio, mean_position_tic, mean_transactions

    def step(self, actions):
        self._terminal = (self._time_index >= len(self._sorted_times) - 2)

        if self._terminal and self._verbose >= 1:
            profit, max_draw_down, sharpe_ratio, fee_ratio, mean_position_tic, mean_transactions = self.get_terminal_stats()
            print('\n=================================')
            print('Initial portfolio value:{}'.format(self._asset_memory['final'][0]))
            print(f'Final portfolio value: {self._portfolio_value}')
            print(f'Final accumulative portfolio value: {profit}')
            print(f'Maximum DrawDown: {max_draw_down}')
            print(f'Sharpe ratio: {sharpe_ratio}')
            print(f'Commission paid: {self.commission_paid}')
            print(f'Commission ratio: {fee_ratio}')
            print(f'Mean position tic per time: {mean_position_tic}')
            print(f'Mean transactions per time: {mean_transactions}')
            print('=================================')

        # transform action to numpy array (if it's a list)
        # actions = np.array(actions, dtype=np.float32)

        # if necessary, normalize weights
        if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
            weights = actions
        else:
            # print('_custom_softmax')
            # weights = _custom_softmax(actions)
            weights = _softmax(actions)

        # save initial portfolio weights for this time step
        self._actions_memory.append(weights)

        # get last step final weights and portfolio_value
        current_portfolio, fee = self._rebalance_portfolio(weights)  # weights*self._portfolio_value, 1000
        self.commission_paid += fee
        self._portfolio_value = current_portfolio.sum()

        # save initial portfolio value of this time step
        self._asset_memory['initial'].append(self._portfolio_value)

        # load next state
        self._time_index += 1
        # calc _price_variation
        self._state, self._info = self._get_state_and_info_from_time_index(self._time_index)
        self._info['trf_mu'] = 1 - (fee / self._portfolio_value)

        # time passes and time variation changes the portfolio distribution
        new_portfolio = current_portfolio * self._price_variation
        self._portfolio_memory.append(new_portfolio)

        # calculate new portfolio value and weights
        self._portfolio_value = np.sum(new_portfolio)
        final_weights = new_portfolio / self._portfolio_value
        self._info['real_weights'] = final_weights
        self._info['portfolio_dist'] = final_weights

        # save final portfolio value and weights of this time step
        self._asset_memory['final'].append(self._portfolio_value)
        self._final_weights.append(final_weights)

        # save date memory
        self._date_memory.append(self._info['end_time'])

        # define portfolio return
        rate_of_return = self._asset_memory['final'][-1] / self._asset_memory['final'][-2]
        portfolio_return = rate_of_return - 1
        mean_return = self._mean_temporal_variation[self._sorted_times[self._time_index]] - 1

        # save portfolio return memory
        self._portfolio_return_memory.append(portfolio_return)

        # Define portfolio return
        # self._reward = math.log(rate_of_return) * self._reward_scaling
        self._reward = (portfolio_return - mean_return) * self._reward_scaling

        return self._state, self._reward, self._terminal, False, {}

    def eval_trades(self, rest_cash):
        tic_counts = self._tic_counts_memory[-1]
        diff_tic_counts = tic_counts - self._tic_counts_memory[-2]
        date = self._sorted_times[self._time_index]
        transactions = np.nonzero(diff_tic_counts)[0]
        tics = self._df['tic'].unique()
        for idx in transactions:
            elem = diff_tic_counts[idx]
            self.num_of_transactions += 1
            if self._verbose > 1:
                if elem > 0:
                    print(f'{date}: куплено {elem} акций {tics[idx]}')
                elif elem < 0:
                    print(f'{date}: продано {-elem} акций {tics[idx]}')
        if self._verbose > 1 and len(transactions) > 0:
            print(f'Открыто {np.count_nonzero(tic_counts)} позиций:')
            for idx in np.nonzero(tic_counts)[0]:
                elem = tic_counts[idx]
                print(f'{tics[idx]}={elem}', end=', ')
            print(f'\nСвободных денег: {rest_cash}')

    def _rebalance_portfolio(self, weights):
        """
        Перераспределяет денежные средства на основе новых весов портфеля и лотности акций.

        :param weights: одномерный массив NumPy, где 0-й элемент — это доля денежных средств,
                        а остальные элементы — это доля по каждой акции.

        :return: Новая распределённая стоимость портфеля (на основе self._portfolio_value) + уже учтённая в нём комиссия
        """
        wsum = weights.sum()
        if wsum > 1:
            # print('bad weights sum', weights.sum())
            weights /= wsum
        data_slice = self._df[
            (self._df[self._time_column] == self._sorted_times[self._time_index])
        ].set_index('tic')

        # В начале - денежные средства
        price = np.array([1.0] + data_slice[self._valuation_column].tolist())
        lot_size = np.array([1.0] + data_slice[self._lot_column].tolist())  # Рубли можем дробить на копейки
        multipliers = price * lot_size

        # Смысл цикла в том, что сперва пытаемся распределить портфель вообще без резервирования денег на комиссию.
        # Такая ситуация допустима, если веса портфеля не менялись.
        # После первой итерации fee содержит необходимые средства для перебалансировки (комиссию),
        # их резервируем и запускаем балансировку ещё раз, т.к. доступные средства уменьшились.
        # В большинстве случаев распределение меньшей суммы денег съедает меньше комиссии, но не всегда,
        # поэтому не можем гарантировать, что цикл завершится за 2 итерации.
        fee = 0
        counter = 0
        while True:
            counter += 1
            if counter > 2:
                print(f'Warning: too much rebalancing ({counter}).')
            real_values, fee, tic_counts = self._allocate_with_fee(price, weights, multipliers, fee)
            rest_cash = real_values[0] - fee
            if rest_cash >= 0:
                break

        self._tic_counts_memory.append(tic_counts)
        self.eval_trades(rest_cash)

        real_values[0] = rest_cash
        return real_values, fee

    def _allocate_with_fee(self, price, weights, multipliers, reserved):
        tic_price = price[1:]
        value_before = self._portfolio_value

        to_distribute = self._portfolio_value - reserved
        real_values = discrete_allocation_custom(weights, multipliers, to_distribute)
        if real_values.sum() > to_distribute:
            raise ValueError('Error discrete allocation.')

        # Считаем количество тикеров (без учёта кеша)
        tic_counts = np.array([int(np.round(x)) for x in (real_values[1:] / tic_price)])
        diff_tic_counts = tic_counts - self._tic_counts_memory[-1]

        # Откатываем незначительные транзакции (менее 1% портфеля) для минимизации комиссии и стабилизации агента
        threshold = real_values.sum() * 0.01
        new_diff_tic_counts = minimize_transactions(tic_price, diff_tic_counts, threshold, real_values[0])
        new_tic_counts = self._tic_counts_memory[-1] + new_diff_tic_counts
        new_real_values = tic_price * new_tic_counts
        if new_real_values.sum() > value_before:
            raise ValueError(f'Distributed more ({new_real_values.sum()}) than necessary ({value_before})')
        new_real_values = np.insert(new_real_values, 0, value_before - new_real_values.sum())

        # Расчёт комиссии за сделки
        turnover = (np.abs(new_diff_tic_counts) * tic_price).sum()
        fee = turnover * self._comission_fee_pct

        if abs(value_before - new_real_values.sum()) > 1e-5:
            raise ValueError(
                f'Portfolio size is not the same before ({value_before}) and after ({new_real_values.sum()}) allocation.')
        return new_real_values, fee, new_tic_counts

    def reset(self, seed=None, options=None):
        self.commission_paid = 0
        self.num_of_transactions = 0
        self._reward = 0
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state, self._info = self._get_state_and_info_from_time_index(self._time_index)
        self._portfolio_value = self._initial_amount
        self._terminal = False

        return self._state, {}

    def _get_state_and_info_from_time_index(self, time_index):
        # Определение временного диапазона
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

        # Выбор данных для текущего временного шага
        self._data = self._df[
            (self._df[self._time_column] >= start_time) &
            (self._df[self._time_column] <= end_time)]

        # Определение вариации цены для текущего временного шага
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self._time_column] == end_time
            ][self._valuation_column].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)  # Добавляем кеш

        # Получение данных по тикерам
        all_tic_data = self._data.set_index(self._tic_column).loc[self._tic_list, self._window_features].to_numpy()
        all_tic_data = all_tic_data.reshape(len(self._tic_list), len(self._window_features), self._time_window)

        # Формирование price_data (транспонирование для приведения к нужному формату)
        # (features, tics, timesteps)
        price_data = all_tic_data.transpose(1, 0, 2)

        # Извлечение индикаторов для последней даты в окне (end_time) по каждому тикеру
        last_time_data = self._data[self._data[self._time_column] == end_time]
        indicators = last_time_data.set_index(self._tic_column).loc[
            self._tic_list, self._indicators_features].to_numpy()

        # Извлечение общих данных по рынку
        # Общие данные предполагаются одинаковыми для всех тикеров
        common_data = last_time_data[self._time_features].iloc[0].to_numpy()

        # Формирование словаря состояния
        state = {
            'price_data': price_data,  # (features, tics, timesteps)
            'indicators': indicators,  # (tic_count, indicators_num)
            'common_data': common_data,  # (common_num,)
            'portfolio_dist': self._final_weights[-1]  # (tic_count + 1,)
        }

        # Формирование дополнительной информации
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

    def _preprocess_data(self):
        # order time dataframe by tic and time
        self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df()[
            [self._tic_column, self._time_column, self._valuation_column]
        ]
        self._mean_temporal_variation = (self._df_price_variation
                                         .groupby(self._time_column)[self._valuation_column]
                                         .mean())
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(self._df_price_variation[self._time_column])
        self._mean_temporal_variation.index = pd.to_datetime(self._mean_temporal_variation.index)

    def _reset_memory(self):
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
        self._portfolio_memory = [
            np.array([self._initial_amount] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        self._tic_counts_memory = [
            np.array([0] * self.portfolio_size, dtype=np.uint32)
        ]
        # memorize datetimes
        date_time = self._sorted_times[self._time_index]
        self._date_memory = [date_time]

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
        numeric_cols = df_temporal_variation.select_dtypes(include=[np.number]).columns
        shifted_data = {}

        for column in numeric_cols:
            shifted_data[f'prev_{column}'] = (
                df_temporal_variation.groupby(self._tic_column)[column]
                .shift(periods=periods)
            )

        # Конкатенируем сдвинутые столбцы с основным DataFrame
        df_temporal_variation = pd.concat([df_temporal_variation, pd.DataFrame(shifted_data)], axis=1)

        # Вычисляем темпоральное изменение и удаляем временные столбцы
        for column in numeric_cols:
            prev_column = f'prev_{column}'
            df_temporal_variation[column] = df_temporal_variation[column] / df_temporal_variation[prev_column]

        # Удаляем временные столбцы и заменяем бесконечности и NaN на 1
        df_temporal_variation = (
            df_temporal_variation
            .drop(columns=list(shifted_data.keys()))
            .replace([float('inf'), -float('inf')], 1)
            .fillna(1)
            .reset_index(drop=True)
        )
        return df_temporal_variation

    def get_portfolio_size_history(self):
        return self._asset_memory['final']
