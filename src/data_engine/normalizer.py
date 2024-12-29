import pickle

import numpy as np


def normalize_with_stats(data, stats=None, exclude=('lot', 'price')):
    df = data.copy()
    df['price'] = df['close'].copy()
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in exclude]

    if stats is not None:
        # Если статистики заданы, проверяем их на совместимость с текущим набором данных
        expected_columns = set(numeric_cols)
        stats_log_cols = set(stats.get('log_cols', []))
        stats_standardize_cols = set(stats.get('standardize', {}).keys())

        if expected_columns != (stats_log_cols | stats_standardize_cols):
            raise ValueError(
                "Несовместимые статистики: отсутствуют необходимые столбцы или присутствуют лишние столбцы.")

        # Использование логарифмической нормализации по сохранённым статистикам
        for col in stats['log_cols']:
            df[col] = np.log(df[col] + 1)

        # Стандартизация по сохранённым средним и стандартным отклонениям
        for col, (mean, std) in stats['standardize'].items():
            df[col] = (df[col] - mean) / std

        return df, stats

    # Если статистики не заданы, вычисляем их
    stats = {'log_cols': [], 'standardize': {}}

    # Выбор столбцов для логарифмической нормализации на основе близкого максимального значения
    max_close = df['price'].max()
    log_cols = [col for col in numeric_cols
                if df[col].min() >= 0
                and df[col].max() > max_close * 0.9]
    stats['log_cols'] = log_cols

    for col in log_cols:
        df[col] = np.log(df[col] + 1)

    # Стандартизация всех числовых столбцов и сохранение статистик
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        stats['standardize'][col] = (mean, std)
        df[col] = (df[col] - mean) / std

    return df, stats


def save_stats(stats, filename):
    with open(filename, 'wb') as f:
        pickle.dump(stats, f)


def load_stats(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
