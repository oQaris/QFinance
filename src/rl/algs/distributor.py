import unittest

import numpy as np


def distribute_optimally(weights, multipliers, total):
    # Шаг 1: начальное распределение (аппроксимация)
    raw_result = weights * total
    result = np.round(raw_result / multipliers) * multipliers
    current_sum = np.sum(result)
    diffs = np.abs((result / multipliers) / total - weights)

    # Шаг 2: корректировка результата
    while current_sum > total:
        # Уменьшаем наиболее значительное значение, если оно может быть уменьшено
        valid_indices = np.where(result - multipliers >= 0)[0]
        if len(valid_indices) == 0:
            return result
        min_diff_idx = valid_indices[np.argmin(diffs[valid_indices])]
        result[min_diff_idx] -= multipliers[min_diff_idx]
        current_sum -= multipliers[min_diff_idx]
        diffs[min_diff_idx] = abs(
            (result[min_diff_idx] / multipliers[min_diff_idx]) / total - weights[min_diff_idx])

    while current_sum < total:
        # Увеличиваем наименьшее значение
        min_diff_idx = np.argmin(diffs)
        if current_sum + multipliers[min_diff_idx] <= total:
            result[min_diff_idx] += multipliers[min_diff_idx]
            current_sum += multipliers[min_diff_idx] # Не пересчитываем сумму в целях оптимизации
            diffs[min_diff_idx] = abs(
                (result[min_diff_idx] / multipliers[min_diff_idx]) / total - weights[min_diff_idx])
        else:
            break

    return result


class TestDistributeOptimally(unittest.TestCase):

    def test_full_sum_for_2(self):
        weights = np.array([0.6, 0.4])
        multipliers = np.array([2., 3.])
        total = 120.

        actual = distribute_optimally(weights, multipliers, total)
        expected = np.array([72, 48])  # 72/120 = 0.6, 48/120 = 0.4

        np.testing.assert_array_equal(actual, expected)

    def test_full_sum_for_3(self):
        weights = np.array([0.2, 0.3, 0.5])
        multipliers = np.array([100., 5., 10.])
        total = 900.

        actual = distribute_optimally(weights, multipliers, total)
        expected = np.array([100, 270, 450])  # 100 + 270 + 450 = 900

        np.testing.assert_array_equal(actual, expected)

    def test_full_sum_for_4(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        multipliers = np.array([5., 10., 20., 25.])
        total = 500.

        actual = distribute_optimally(weights, multipliers, total)
        expected = np.array([40, 100, 160, 200])

        np.testing.assert_array_equal(actual, expected)

    def test_partial_sum_total_1(self):
        weights = np.array([0.5, 0.3, 0.2])
        multipliers = np.array([1., 2., 3.])
        total = 1.

        actual = distribute_optimally(weights, multipliers, total)
        expected = np.array([0, 0, 0])

        np.testing.assert_array_equal(actual, expected)

    def test_partial_sum_total_12(self):
        weights = np.array([0.5, 0.3, 0.2])
        multipliers = np.array([3., 2., 1.])
        total = 12.

        actual = distribute_optimally(weights, multipliers, total)
        expected = np.array([6, 4, 2])

        np.testing.assert_array_equal(actual, expected)

    def test_zero_multiplier(self):
        weights = np.array([1.0, 5.0, 2.6])
        multipliers = np.array([0.3, 2.0, 0.])
        total = 100.

        actual = distribute_optimally(weights, multipliers, total)
        expected = np.array([99.9, 500, np.nan])

        np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=1e-5, atol=1e-8)
        # Проверяем отдельно, что последний элемент равен np.nan
        self.assertTrue(np.isnan(actual[-1]) and np.isnan(expected[-1]))
