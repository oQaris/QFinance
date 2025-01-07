import numpy as np
import pandas as pd


class StockQuoteValidator:
    def __init__(self, threshold: float = 0.9, row_count: int = 5):
        """
        Инициализация валидатора котировок.

        :param threshold: Порог изменения цен для обнаружения скачков (по умолчанию 90%)
        :param row_count: Количество примеров, выводимых в случае ошибки (по умолчанию 5)
        """
        self.threshold = threshold
        self.row_count = row_count

    def validate(self, df: pd.DataFrame) -> list:
        """
        Валидирует DataFrame с котировками.

        :param df: DataFrame с колонками: 'date', 'tic', 'open', 'close', 'high', 'low', 'volume', и, возможно, 'lot'
        :return: Список ошибок или сообщение об успехе.
        """
        errors = []

        # Определяем числовые колонки для проверки
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 0) Проверка на отсутствие NaN и бесконечностей
        if df[numeric_cols].isnull().values.any():
            missing_info = df[df[numeric_cols].isnull().any(axis=1)].sample(n=min(self.row_count, len(df)))
            errors.append(f"Есть пустые значения (NaN):\n{missing_info}.")

        if np.isinf(df[numeric_cols].to_numpy()).any():
            inf_info = (df[df[numeric_cols].apply(lambda x: np.isinf(x)).any(axis=1)]
                        .sample(n=min(self.row_count, len(df))))
            errors.append(f"Есть бесконечные значения:\n{inf_info}.")

        # 1) Проверка high > low
        invalid_high_low_info = df[df['high'] < df['low']]
        if not invalid_high_low_info.empty:
            invalid_high_low_info = invalid_high_low_info.sample(n=min(self.row_count, len(invalid_high_low_info)))
            errors.append(f"Некорректные значения high < low:\n{invalid_high_low_info}.")

        # 2) Проверка на положительные значения в 'open', 'close', 'high', 'low' и, если есть, 'lot',
        # при этом 'volume' может быть нулевым, но не отрицательным
        cols_to_check = ['open', 'close', 'high', 'low']
        if 'lot' in df.columns:
            cols_to_check.append('lot')

        # Проверяем, что значения в указанных столбцах положительные, а volume неотрицательный
        invalid_non_positive_info = df[(df[cols_to_check] <= 0).any(axis=1) | (df['volume'] < 0)]
        if not invalid_non_positive_info.empty:
            invalid_non_positive_info = \
                invalid_non_positive_info.sample(n=min(self.row_count, len(invalid_non_positive_info)))
            errors.append(
                f"Некорректные значения в столбцах {cols_to_check} (должны быть > 0) и volume (не должно быть < 0):\n{invalid_non_positive_info}")

        # 3) Проверка open и close в пределах high и low
        invalid_open_info = df[(df['open'] > df['high']) | (df['open'] < df['low'])]
        if not invalid_open_info.empty:
            invalid_open_info = invalid_open_info.sample(n=min(self.row_count, len(invalid_open_info)))
            errors.append(f"Некорректные значения open:\n{invalid_open_info}.")

        invalid_close_info = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
        if not invalid_close_info.empty:
            invalid_close_info = invalid_close_info.sample(n=min(self.row_count, len(invalid_close_info)))
            errors.append(f"Некорректные значения close:\n{invalid_close_info}.")

        # 4) Проверка одинакового набора дат для каждой группы tic
        unique_dates_per_tic = df.groupby('tic')['date'].nunique()
        if unique_dates_per_tic.nunique() != 1:
            errors.append("Набор дат не одинаков для всех тикеров.")

        # 5) Проверка на дубликаты дат
        duplicated_dates_info = df[df.duplicated(subset=['tic', 'date'])]
        if not duplicated_dates_info.empty:
            duplicated_dates_info = duplicated_dates_info.sample(n=min(self.row_count, len(duplicated_dates_info)))
            errors.append(f"Обнаружены дубликаты дат:\n{duplicated_dates_info}.")

        # 6) Проверка на порядок дат внутри каждой группы 'tic'
        tics_with_wrong_dates = []
        wrong_date_samples = []
        for tic, group in df.groupby('tic'):
            if not group['date'].is_monotonic_increasing:
                tics_with_wrong_dates.append(tic)
                wrong_date_samples.append(group[['tic', 'date']].sample(n=min(self.row_count, len(group))))
        if tics_with_wrong_dates:
            errors.append(
                f"Некоторые даты не отсортированы по возрастанию для тикеров: {', '.join(tics_with_wrong_dates)}")
            errors.append(f"Примеры неверных дат:\n{pd.concat(wrong_date_samples)}")

        # todo Проверка взаимосвязи между open и close (цена открытия текущего дня должна совпадать с закрытием предыдущего)

        return errors if errors else ["Все проверки пройдены."]
