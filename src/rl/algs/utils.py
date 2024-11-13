import pandas as pd


def calculate_periods_per_year(data):
    tickers = data['tic'].unique()
    periods_list = []

    for tic in tickers:
        tic_data = data[data['tic'] == tic].copy()
        tic_data = tic_data.sort_values(by='date')

        # Вычисляем среднюю разницу между датами
        avg_diff = tic_data['date'].diff().dropna().mean()

        # Рассчитываем количество периодов в году, учитывая високосные
        periods_per_year = pd.Timedelta(days=365.25) / avg_diff
        periods_list.append(periods_per_year)

    return sum(periods_list) / len(periods_list)


if __name__ == '__main__':
    # Пример использования
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='15min').tolist() * 2,  # 15-минутные интервалы
        'tic': ['AAPL'] * 34945 + ['MSFT'] * 34945  # Пример данных
    })

    periods = calculate_periods_per_year(df)
    print("Количество периодов в году:", periods)
