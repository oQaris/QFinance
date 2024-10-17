import unittest

import numpy as np
import pandas as pd
from pypfopt import DiscreteAllocation


def discrete_allocation(weights, multipliers, total):
    """
     Перераспределяет сумму total по ячейкам массива так,
     чтобы распределение было максимально близко к заданным весам weights.
     То есть чтобы sum(abs(result[i] / total - weight[i])) было минимальным.
     При этом значения в результирующем массиве кратны соответствующим значениям из массива кратности,
     то есть all(result[i] % distr[i] == 0).
    :param weights: Массив долей тикеров в портфеле, состоит из чисел от 0 до 1, сумма которых сходится в 1.
    :param multipliers: Массив лотности каждого тикера, состоит из целых чисел больше 0.
    :param total: Сумма для перераспределения.
    :return: Массив result, сумма значений которого меньше либо равна total.
    """
    # Преобразуем weights и multipliers в нужные форматы
    tickers = np.arange(len(weights))  # создаем фиктивные тикеры на основе индексов
    weights_dict = {ticker: weight for ticker, weight in zip(tickers, weights)}
    multipliers_series = pd.Series(multipliers, index=tickers)

    # Выполняем дискретное распределение библиотекой pypfopt
    d = DiscreteAllocation(weights_dict, multipliers_series, total_portfolio_value=total, short_ratio=0)
    allocation, rem = d.greedy_portfolio(reinvest=False)

    # Преобразуем allocation обратно в np.array
    allocation_array = np.zeros(len(weights))
    for ticker, alloc in allocation.items():
        allocation_array[ticker] = alloc

    return allocation_array * multipliers


class TestDiscreteAllocation(unittest.TestCase):

    def test_full_sum_for_2(self):
        weights = np.array([0.6, 0.4])
        multipliers = np.array([2., 3.])
        total = 120.

        actual = discrete_allocation(weights, multipliers, total)
        expected = np.array([72, 48])  # 72/120 = 0.6, 48/120 = 0.4

        np.testing.assert_array_equal(actual, expected)

    def test_full_sum_for_3(self):
        weights = np.array([0.2, 0.3, 0.5])
        multipliers = np.array([100., 5., 10.])
        total = 900.

        actual = discrete_allocation(weights, multipliers, total)
        expected = np.array([100, 270, 450])  # 100 + 270 + 450 = 900

        np.testing.assert_array_equal(actual, expected)

    def test_full_sum_for_4(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        multipliers = np.array([5., 10., 20., 25.])
        total = 500.

        actual = discrete_allocation(weights, multipliers, total)
        expected = np.array([50, 100, 140, 200])

        np.testing.assert_array_equal(actual, expected)

    def test_partial_sum_total_1(self):
        weights = np.array([0.5, 0.3, 0.2])
        multipliers = np.array([1., 2., 3.])
        total = 1.

        actual = discrete_allocation(weights, multipliers, total)
        expected = np.array([1, 0, 0])

        np.testing.assert_array_equal(actual, expected)

    def test_partial_sum_total_12(self):
        weights = np.array([0.5, 0.3, 0.2])
        multipliers = np.array([3., 2., 1.])
        total = 12.

        actual = discrete_allocation(weights, multipliers, total)
        expected = np.array([6, 4, 2])

        np.testing.assert_array_equal(actual, expected)

    def test_zero_multiplier(self):
        weights = np.array([1.0, 5.0, 2.6])
        multipliers = np.array([0.3, 2.0, 0.])
        total = 100.

        # Ожидаем, что будет выброшено исключение, например, ValueError
        with self.assertRaises(AssertionError):
            discrete_allocation(weights, multipliers, total)
