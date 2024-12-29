import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs

# см. значение за год - https://cbr.ru/hd_base/zcyc_params/zcyc/
yearly_risk_free_rate = 0.2268


def sharpe_sortino(df):
    # Вычисляем среднюю разницу между датами
    avg_diff = pd.to_datetime(df['date']).diff().dropna().mean()
    # Рассчитываем количество периодов в году, учитывая високосные
    periods_per_year = pd.Timedelta(days=365.25) / avg_diff
    args = {
        'rf': yearly_risk_free_rate,
        'periods': periods_per_year,
        'annualize': True,
        'smart': True
    }
    sharpe_ratio = qs.stats.sharpe(df['returns'], **args)
    sortino_ratio = qs.stats.sortino(df['returns'], **args)
    return sharpe_ratio, sortino_ratio


def calculate_periods_per_year(df):
    tickers = df['tic'].unique()
    periods_list = []

    for tic in tickers:
        tic_data = df[df['tic'] == tic].copy()
        tic_data = tic_data.sort_values(by='date')

        # Вычисляем среднюю разницу между датами
        avg_diff = pd.to_datetime(tic_data['date']).diff().dropna().mean()

        # Рассчитываем количество периодов в году, учитывая високосные
        periods_per_year = pd.Timedelta(days=365.25) / avg_diff
        periods_list.append(periods_per_year)

    return sum(periods_list) / len(periods_list)


def calculate_equal_weight_portfolio(initial_amount, mean_temporal_variation):
    """
    Рассчитывает стоимость портфеля с равномерным распределением акций.
    """
    equal_weight_portfolio = [initial_amount]

    for variation in mean_temporal_variation[1:-1]:
        equal_weight_portfolio.append(equal_weight_portfolio[-1] * variation)

    return equal_weight_portfolio


def plot_with_risk_free(portfolio, n_periods_per_year, rf_rate=yearly_risk_free_rate, equal_weight_portfolio=None):
    """
    Рисует график с кривыми стратегии, безубыточного актива и равномерного распределенного портфеля.
    """
    len_trade = len(portfolio)
    daily_rf_rate = (1 + rf_rate) ** (1 / n_periods_per_year) - 1
    risk_free_portfolio = portfolio[0] * (1 + daily_rf_rate) ** np.arange(len_trade)

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio, label="Стратегия", color="green")
    plt.plot(risk_free_portfolio, label=f"Безубыточный актив ({rf_rate * 100:.2f}% годовых)", color="blue",
             linestyle="--")

    if equal_weight_portfolio is not None:
        plt.plot(equal_weight_portfolio, label="Равномерно распределенный портфель", color="orange", linestyle=":")

    plt.title("Изменение размера портфеля", fontsize=16)
    plt.xlabel("Периоды", fontsize=14)
    plt.ylabel("Размер портфеля", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == '__main__':
    # Пример использования
    date_list = pd.date_range(start='2023-01-01', end='2023-12-31', freq='15min').tolist()
    test_df = pd.DataFrame({
        'date': date_list * 2,  # интервалы
        'tic': ['AAPL'] * len(date_list) + ['MSFT'] * len(date_list)  # Пример данных
    })

    test_periods = calculate_periods_per_year(test_df)
    print("Количество периодов в году:", test_periods)

    test_df = pd.DataFrame({
        'date': date_list,
        'returns': pd.Series(np.random.normal(0.00001, 0.0025, len(date_list)))
    })
    test_sharpe_ratio, test_sortino_ratio = sharpe_sortino(test_df)
    print("sharpe:", test_sharpe_ratio)
    print("sortino:", test_sortino_ratio)

    initial_capital = 100
    test_portfolio = initial_capital * (1 + test_df['returns']).cumprod()

    # Построение графика
    plot_with_risk_free(test_portfolio, test_periods)
