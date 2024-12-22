from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
import pandas as pd
import quantstats as qs
from gymnasium import spaces

from src.rl.algs import utils
from src.rl.algs.distributor import discrete_allocation_custom, minimize_transactions
from src.rl.algs.utils import plot_with_risk_free, calculate_periods_per_year


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


def scale_to_unit_sum(arr):
    """
    Масштабирует массив NumPy в диапазон [0, 1] так, чтобы сумма элементов была равна 1.
    Корректно работает с отрицательными значениями и типами float32.
    """
    arr = arr.astype(np.float64)  # Приводим к float64 для избежания переполнения

    arr_min = np.min(arr)
    arr_max = np.max(arr)

    if arr_max == arr_min:  # Если все элементы одинаковы
        return np.ones_like(arr, dtype=np.float32) / arr.size

    # Линейное масштабирование значений в диапазон [0, 1]
    scaled = (arr - arr_min) / (arr_max - arr_min)

    # Нормализация так, чтобы сумма была равна 1
    result = scaled / np.sum(scaled)

    return result.astype(np.float32)  # Возвращаем в исходный тип float32


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
            reward_type='excess_absolute',
            reward_scaling=10,
            fee_ratio=0.003,
            transaction_threshold_ratio=0.01,
            verbose=0,
    ):
        """
        Среда портфельной аллокации, в которой наблюдение - Dict,
        действие - Box (текущее распределение стоимости портфеля).

        :param df: Датафрейм с историческими данными для среды
        :param initial_amount: Начальная стоимость портфеля (значение кеша)
        :param window_features: Столбцы временного ряда (изменения цен)
        :param indicators_features: Столбцы индикаторов активов
        :param time_features: Столбцы общих данных по дате (настроение рынка, турбулентность, данные о времени)
        :param valuation_column: Столбец реальной цены для вычисления наград
        :param time_column: Столбец метки времени
        :param tic_column: Столбец названия тикера
        :param lot_column: Столбец лотности актива
        :param time_window: Размер временного окна 'time_features' для добавления в словарь состояния
        :param reward_type: Тип вычисления награды агента.
            'return_absolute' — процентная доходность,
            'return_log' — логарифмическая доходность,
            'excess_absolute' — разница со средним движением рынка,
            'excess_relative' — отношение со средним движением рынка,
            'excess_log' — логарифм от excess_relative.
        :param reward_scaling: Множитель награды агента
        :param fee_ratio: Доля комисси за сделки (от стоимости транзакции)
        :param transaction_threshold_ratio: Порог отсечения транзакций (в долях от стоимости портфеля)
        :param verbose: Уровень детализации оповещений.
            0 — отсутствие вывода,
            1 — информационные сообщения,
            2 — отладочные сообщения
        """
        self.commission_paid = 0
        self.num_of_transactions = 0
        self.time_window = time_window
        self._time_index = time_window - 1

        self.valuation_column = valuation_column
        self.time_column = time_column
        self.tic_column = tic_column
        self.lot_column = lot_column

        self._df = df
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.reward_type = reward_type
        self.fee_ratio = fee_ratio
        self.transaction_threshold_ratio = transaction_threshold_ratio

        self.window_features = window_features
        self.indicators_features = indicators_features
        self.time_features = time_features

        self.verbose = verbose
        # initialize price variation
        self._df_price_variation = None
        # preprocess data
        self._preprocess_data()
        # dims and spaces
        self._tic_list = self._df[self.tic_column].unique()
        self.portfolio_size = len(self._tic_list)
        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        self._portfolio_value = self.initial_amount

        # Необходимо для OpenAI Gym
        max_float = 1.7 * 10308
        self.action_space = spaces.Box(low=-max_float, high=max_float, shape=(self.portfolio_size + 1,),
                                       dtype=np.float32)
        self.observation_space = spaces.Dict({
            'price_data': spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.window_features), self.portfolio_size, self.time_window),
                dtype=np.float32
            ),
            'indicators': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.portfolio_size, len(self.indicators_features)), dtype=np.float32
            ),
            'common_data': spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.time_features),), dtype=np.float32
            ),
            'portfolio_dist': spaces.Box(
                low=0, high=1, shape=(self.portfolio_size + 1,), dtype=np.float32
            )
        })
        self.metadata = {'render_modes': ['human'], 'render_fps': 1}
        self.render_mode = 'human'
        self._reset_memory()

    def is_pred_terminal_state(self):
        return self._time_index >= len(self._sorted_times) - 3

    def get_terminal_stats(self):
        metrics_df = pd.DataFrame({
            'date': self._date_memory,
            'returns': self._portfolio_return_memory,
            'portfolio_values': self._asset_memory,
            'tic_counts': self._tic_counts_memory
        })

        # Считаем среднее число транзакций без учёта первых покупок
        final_num_transactions = self.num_of_transactions - np.count_nonzero(self._tic_counts_memory[0])
        mean_transactions = final_num_transactions / (len(self._tic_counts_memory) - 1)

        # Продаём всё, перед тем как считать профит
        sell_all_weights = np.array([1] + [0] * self.portfolio_size)
        final_portfolio_value = self._rebalance_portfolio(sell_all_weights)[0].sum()
        profit = final_portfolio_value / self.initial_amount

        # Берёт с отрицанием, чтобы просадка была положительной
        max_draw_down = -qs.stats.max_drawdown(metrics_df['portfolio_values'])
        sharpe_ratio, sortino_ratio = utils.sharpe_sortino(metrics_df)
        fee_ratio = self.commission_paid / self.initial_amount

        # Первый элемент не учитываем, т.к. там нули
        positions = np.stack(metrics_df['tic_counts'].values[1:])
        mean_position_tic = np.count_nonzero(positions, axis=1).mean()

        return {
            'profit': profit,
            'max_draw_down': max_draw_down,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'fee_ratio': fee_ratio,
            'mean_position_tic': mean_position_tic,
            'mean_transactions': mean_transactions,
        }

    def step(self, actions):
        terminal = self.is_terminal_state()
        info = {}

        # transform action to numpy array (if it's a list)
        # actions = np.array(actions, dtype=np.float32)

        # if necessary, normalize weights
        if np.abs(np.sum(actions) - 1) < 1e-4 and np.min(actions) >= 0:
            weights = actions / np.sum(actions)
        else:
            # print('_custom_softmax')
            # weights = _custom_softmax(actions)
            # weights = _softmax(actions)
            weights = scale_to_unit_sum(actions)

        # save initial portfolio weights for this time step
        self._actions_memory.append(weights)

        # get last step final weights and portfolio_value
        current_portfolio, fee = self._rebalance_portfolio(weights)  # portfolio = weights * self._portfolio_value
        self.commission_paid += fee
        self._portfolio_value = current_portfolio.sum()

        # load next state
        self._time_index += 1
        # calc _price_variation
        state = self._get_state_and_info_from_time_index(self._time_index)

        # time passes and time variation changes the portfolio distribution
        new_portfolio = current_portfolio * self._price_variation
        self._portfolio_memory.append(new_portfolio)

        # calculate new portfolio value and weights
        self._portfolio_value = np.sum(new_portfolio)
        final_weights = new_portfolio / self._portfolio_value
        state['portfolio_dist'] = final_weights.astype(np.float32)

        # save final portfolio value and weights of this time step
        self._asset_memory.append(self._portfolio_value)
        self._final_weights.append(final_weights)

        # save date memory
        end_time = self._sorted_times[self._time_index]
        self._date_memory.append(end_time)

        # define portfolio return
        rate_of_return = self._asset_memory[-1] / self._asset_memory[-2]
        portfolio_return = rate_of_return - 1
        rate_of_mean_return = self._mean_temporal_variation[self._sorted_times[self._time_index]]
        mean_return = rate_of_mean_return - 1

        # save portfolio return memory
        self._portfolio_return_memory.append(portfolio_return)

        # define reward
        if self.reward_type == 'return_absolute':
            # Процентная доходность
            reward = portfolio_return
        elif self.reward_type == 'return_log':
            # Логарифмическая доходность
            reward = math.log(rate_of_return)
        elif self.reward_type == 'excess_absolute':
            # Среднерыночная процентная доходность
            reward = portfolio_return - mean_return
        elif self.reward_type == 'excess_relative':
            # Избыточная доходность
            reward = rate_of_return / rate_of_mean_return
        elif self.reward_type == 'excess_log':
            # Логарифм избыточной доходности
            reward = math.log(rate_of_return / rate_of_mean_return)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        if terminal:
            terminal_stats = self.get_terminal_stats()
            info['terminal_stats'] = terminal_stats
            if self.verbose >= 1:
                print('\n=================================')
                print(f'Initial portfolio value:{self.initial_amount}')
                print(f'Final portfolio value: {self._portfolio_value}')
                for key, value in terminal_stats.items():
                    print(f'{key}: {value}')
                print('=================================')

        return state, reward * self.reward_scaling, terminal, False, info

    def eval_trades(self, rest_cash):
        tic_counts = self._tic_counts_memory[-1]
        diff_tic_counts = tic_counts - self._tic_counts_memory[-2]
        date = self._sorted_times[self._time_index]
        transactions = np.nonzero(diff_tic_counts)[0]
        tics = self._df['tic'].unique()
        for idx in transactions:
            elem = diff_tic_counts[idx]
            self.num_of_transactions += 1
            if self.verbose > 1:
                if elem > 0:
                    print(f'{date}: куплено {elem} акций {tics[idx]}')
                elif elem < 0:
                    print(f'{date}: продано {-elem} акций {tics[idx]}')
        if self.verbose > 1 and len(transactions) > 0:
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
        if abs(wsum - 1) > 1e-5:
            warnings.warn(f'Bad weights sum: {weights.sum()}')
            weights /= wsum

        data_slice = self._df[
            (self._df[self.time_column] == self._sorted_times[self._time_index])
        ].set_index('tic')

        # В начале - денежные средства
        price = np.array([1.0] + data_slice[self.valuation_column].tolist())
        lot_size = np.array([1.0] + data_slice[self.lot_column].tolist())  # Рубли можем дробить на копейки
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
                warnings.warn(f'Too much rebalancing: {counter}')
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
        sum_allocation = real_values.sum()
        if sum_allocation > to_distribute:
            real_values[0] -= sum_allocation - to_distribute
            if real_values[0] < 0:
                raise ValueError(
                    f'Error discrete allocation. {sum_allocation} > {to_distribute} and {real_values[0]} < 0')

        # Считаем количество тикеров (без учёта кеша)
        tic_counts = np.array([int(np.round(x)) for x in (real_values[1:] / tic_price)])
        diff_tic_counts = tic_counts - self._tic_counts_memory[-1]

        # Откатываем незначительные транзакции (менее 0.5% портфеля) для минимизации комиссии и стабилизации агента
        threshold = real_values.sum() * self.transaction_threshold_ratio
        new_diff_tic_counts = minimize_transactions(tic_price, diff_tic_counts, threshold, real_values[0])
        new_tic_counts = self._tic_counts_memory[-1] + new_diff_tic_counts
        new_real_values = tic_price * new_tic_counts
        if new_real_values.sum() > value_before:
            raise ValueError(f'Distributed more ({new_real_values.sum()}) than necessary ({value_before})')
        new_real_values = np.insert(new_real_values, 0, value_before - new_real_values.sum())

        # Расчёт комиссии за сделки
        turnover = (np.abs(new_diff_tic_counts) * tic_price).sum()
        fee = turnover * self.fee_ratio

        if abs(value_before - new_real_values.sum()) > 1e-5:
            raise ValueError(
                f'Portfolio size is not the same before ({value_before}) and after ({new_real_values.sum()}) allocation.')
        return new_real_values, fee, new_tic_counts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.commission_paid = 0
        self.num_of_transactions = 0
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self.time_window - 1
        self._reset_memory()

        state = self._get_state_and_info_from_time_index(self._time_index)
        self._portfolio_value = self.initial_amount

        return state, {}

    def render(self):
        if self.is_terminal_state():
            plot_with_risk_free(self._asset_memory, calculate_periods_per_year(self._df))

    def is_terminal_state(self):
        return self._time_index >= len(self._sorted_times) - 2

    def _get_state_and_info_from_time_index(self, time_index):
        # Определение временного диапазона
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self.time_window - 1)]

        # Выбор данных для текущего временного шага
        self._data = self._df[
            (self._df[self.time_column] >= start_time) &
            (self._df[self.time_column] <= end_time)]

        # Определение вариации цены для текущего временного шага
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self.time_column] == end_time
            ][self.valuation_column].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)  # Добавляем кеш

        # Получение данных по тикерам
        all_tic_data = self._data.set_index(self.tic_column).loc[self._tic_list, self.window_features].to_numpy()
        all_tic_data = all_tic_data.reshape(len(self._tic_list), len(self.window_features), self.time_window)

        # Формирование price_data (транспонирование для приведения к нужному формату)
        # (features, tics, timesteps)
        price_data = all_tic_data.transpose(1, 0, 2)

        # Извлечение индикаторов для последней даты в окне (end_time) по каждому тикеру
        last_time_data = self._data[self._data[self.time_column] == end_time]
        indicators = last_time_data.set_index(self.tic_column).loc[
            self._tic_list, self.indicators_features].to_numpy()

        # Извлечение общих данных по рынку
        # Общие данные предполагаются одинаковыми для всех тикеров
        common_data = last_time_data[self.time_features].iloc[0].to_numpy()

        # Формирование словаря состояния
        state = {
            'price_data': price_data,  # (features, tics, timesteps)
            'indicators': indicators,  # (tic_count, indicators_num)
            'common_data': common_data,  # (common_num,)
            'portfolio_dist': self._final_weights[-1]  # (tic_count + 1,)
        }
        return state

    def _preprocess_data(self):
        # Сортируем датафрейм по тикерам и времени
        self._df = self._df.sort_values(by=[self.tic_column, self.time_column])

        # Определяем изменение цен тикеров с течением времени
        self._df_price_variation = self._temporal_variation_df()[
            [self.tic_column, self.time_column, self.valuation_column]
        ]
        self._mean_temporal_variation = (self._df_price_variation
                                         .groupby(self.time_column)[self.valuation_column]
                                         .mean())
        # Преобразуем даты
        self._df[self.time_column] = pd.to_datetime(self._df[self.time_column])
        self._df_price_variation[self.time_column] = pd.to_datetime(self._df_price_variation[self.time_column])
        self._mean_temporal_variation.index = pd.to_datetime(self._mean_temporal_variation.index)

    def _reset_memory(self):
        self._asset_memory = [self.initial_amount]
        self._portfolio_return_memory = [0]
        # Начальное действие - все деньги в кеше
        self._actions_memory = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float64)
        ]
        # Для внутренних расчётов используем везде 64 битные типы
        # кроме весов портфеля, поскольку они участвуют в 32 битном state
        self._final_weights = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        self._portfolio_memory = [
            np.array([self.initial_amount] + [0] * self.portfolio_size, dtype=np.float64)
        ]
        self._tic_counts_memory = [
            np.array([0] * self.portfolio_size, dtype=np.uint64)
        ]
        date_time = self._sorted_times[self._time_index]
        self._date_memory = [date_time]

    def _temporal_variation_df(self, periods=1):
        """
        Вычисляет кадры данных временных вариаций для числовых столбцов.

        :param periods: Количество периодов для расчета временных изменений.
        :return: Кадры данных временных вариаций.
        """
        df_temporal_variation = self._df.copy()
        numeric_cols = df_temporal_variation.select_dtypes(include=[np.number]).columns
        shifted_data = {}

        for column in numeric_cols:
            shifted_data[f'prev_{column}'] = (
                df_temporal_variation.groupby(self.tic_column)[column]
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
        return self._asset_memory
