import unittest

import numpy as np
import pandas as pd
from pypfopt import DiscreteAllocation


def discrete_allocation_custom(weights: np.ndarray, prices: np.ndarray, total_sum: float) -> np.ndarray:
    # Рассчитываем целевые суммы для каждого актива
    target_allocations = weights * total_sum

    # Количество акций, которые можно купить по целевой аллокации
    num_shares = np.floor(target_allocations / prices)

    # Оставшаяся сумма после покупки целого числа акций
    remaining_sum = total_sum - np.sum(num_shares * prices)

    # Жадно добавляем акции до тех пор, пока не потратим всю оставшуюся сумму
    while remaining_sum > 0:
        # Рассчитываем ошибки для каждого актива между целевой и реальной аллокацией
        allocation_error = (target_allocations - num_shares * prices) / prices

        # Находим индекс актива с максимальной ошибкой
        best = np.argmax(allocation_error)

        # Если можем купить ещё одну акцию этого актива, покупаем
        if prices[best] <= remaining_sum and weights[best] != 0:
            num_shares[best] += 1
            remaining_sum -= prices[best]
        else:
            break

    return num_shares * prices


def minimize_transactions(price: np.ndarray, diff_tic_counts: np.ndarray, min_transaction: float, cash_balance: float):
    """
    Минимизирует количество транзакций при ребалансировке портфеля.

    :param price: np.ndarray, цена каждой акции
    :param diff_tic_counts: np.ndarray, изменение количества акций для каждой позиции (без кеша)
    :param min_transaction: float, минимальная сумма денег для проведения одной транзакции
    :param cash_balance: float, доступная сумма денег вне позиций
    :return: np.ndarray, скорректированные diff_tic_counts
    """
    # Вычисляем стоимость каждой транзакции
    transaction_values = diff_tic_counts * price
    cash_balance += transaction_values.sum()
    # Сортируем транзакции по абсолютной стоимости в порядке убывания
    sorted_indices = np.argsort(-np.abs(transaction_values))
    # Итоговая корректировка портфеля
    adjusted_transactions = np.zeros_like(diff_tic_counts)

    # Перебираем транзакции начиная с самой большой
    for idx in sorted_indices:
        txn_value = transaction_values[idx]
        # Если транзакция переходит пороговое значение, то выполняем её
        if np.abs(txn_value) >= min_transaction:
            adjusted_transactions[idx] = diff_tic_counts[idx]
            cash_balance -= txn_value

    if cash_balance < 0:
        for idx in sorted_indices:
            txn_value = transaction_values[idx]
            # Выбираем наибольшие невыполненные транзакции на продажу, чтобы восполнить баланс
            if txn_value < 0 and adjusted_transactions[idx] == 0:
                adjusted_transactions[idx] = diff_tic_counts[idx]
                cash_balance -= txn_value
                if cash_balance >= 0:
                    break

    if cash_balance < -1e-6:
        raise ValueError(f'Inconsistency in the number of tickers and prices. cash_balance = {cash_balance}')

    return adjusted_transactions


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
    allocation, rem = d.greedy_portfolio(reinvest=False, verbose=False)

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

        actual = discrete_allocation_custom(weights, multipliers, total)
        expected = np.array([72, 48])  # 72/120 = 0.6, 48/120 = 0.4

        np.testing.assert_array_equal(actual, expected)

    def test_full_sum_for_3(self):
        weights = np.array([0.2, 0.3, 0.5])
        multipliers = np.array([100., 5., 10.])
        total = 900.

        actual = discrete_allocation_custom(weights, multipliers, total)
        expected = np.array([100, 270, 450])  # 100 + 270 + 450 = 820

        np.testing.assert_array_equal(actual, expected)

    def test_full_sum_for_4(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        multipliers = np.array([5., 10., 20., 25.])
        total = 500.

        actual = discrete_allocation_custom(weights, multipliers, total)
        expected = np.array([50, 100, 140, 200])

        np.testing.assert_array_equal(actual, expected)

    def test_partial_sum_total_1(self):
        weights = np.array([0.5, 0.3, 0.2])
        multipliers = np.array([1., 2., 3.])
        total = 1.

        actual = discrete_allocation_custom(weights, multipliers, total)
        expected = np.array([1, 0, 0])

        np.testing.assert_array_equal(actual, expected)

    def test_partial_sum_total_12(self):
        weights = np.array([0.5, 0.3, 0.2])
        multipliers = np.array([3., 2., 1.])
        total = 12.

        actual = discrete_allocation_custom(weights, multipliers, total)
        expected = np.array([6, 4, 2])

        np.testing.assert_array_equal(actual, expected)

    def test_zero_multiplier(self):
        weights = np.array([1.0, 5.0, 2.6])
        multipliers = np.array([0.3, 2.0, 0.])
        total = 100.

        # Ожидаем, что будет выброшено исключение, например, ValueError
        with self.assertRaises(AssertionError):
            discrete_allocation(weights, multipliers, total)
