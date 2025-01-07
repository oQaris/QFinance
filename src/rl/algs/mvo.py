import numpy as np
import pandas as pd
from pypfopt import expected_returns, BlackLittermanModel
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from src.rl.traint_test.env_builder import load_datasets


def modern_portfolio_theory(df):
    df_stocks = _prepare_df(df)

    # Годовая доходность
    mu = expected_returns.mean_historical_return(df_stocks)
    # Дисперсия портфеля
    # sigma = risk_models.sample_cov(df_stocks)
    sigma = risk_models.CovarianceShrinkage(df_stocks).ledoit_wolf()

    # Максимизируем коэффициент Шарпа
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))  # (-1,1), если допустимо торговать в шорт
    ef.max_sharpe()
    sharpe_pwt = ef.clean_weights()
    print(sharpe_pwt)
    ef.portfolio_performance(verbose=True)

    columns = list(df_stocks.columns)
    result = np.zeros(len(columns), dtype=float)
    for index, key in enumerate(columns):
        result[index] = sharpe_pwt[key]
    return result


def black_litterman_theory(df):
    df_stocks = _prepare_df(df)

    # 1. Рассчитаем ковариацию активов и среднюю историческую доходность
    mu = expected_returns.mean_historical_return(df_stocks)
    sigma = risk_models.CovarianceShrinkage(df_stocks).ledoit_wolf()

    # 2. Создаем модель Блэка-Литтермана
    # todo Пустые матрицы для Q и P, их надо эвристически заполнить. Сейчас функция аналогична modern_portfolio_theory()
    q = np.array([])
    p = np.zeros((0, mu.shape[0]))

    bl = BlackLittermanModel(sigma, pi=mu, P=p, Q=q)

    # 3. Получаем скорректированные доходности
    ret_bl = bl.bl_returns()

    # 4. Оптимизация портфеля
    ef = EfficientFrontier(ret_bl, sigma)
    ef.max_sharpe()
    sharpe_pwt = ef.clean_weights()
    # print(sharpe_pwt)
    ef.portfolio_performance(verbose=True)

    columns = list(df_stocks.columns)
    result = np.zeros(len(columns), dtype=float)
    for index, key in enumerate(columns):
        result[index] = sharpe_pwt[key]
    return result


def _prepare_df(df):
    df_stocks = df.copy()
    df_stocks['price'] = df_stocks['price'] * df_stocks['lot']
    df_stocks = df_stocks.pivot_table(
        index='date',
        columns='tic',
        values='price'
    )
    df_stocks['RUB'] = 1
    if df_stocks.isna().sum().sum() > 0:
        raise ValueError('NaN in dataset')
    return df_stocks


def top_stocks(df, n: int, skip_first: int = 0):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(['tic', 'date'])
    # пропускаем первые skip_first записей в каждом tic
    df = df.groupby('tic').apply(lambda x: x.iloc[skip_first:]).reset_index(drop=True)

    # Находим первый и последний закрывающий курс для каждой акции
    grouped = df.groupby('tic').agg(first_price=('price', 'first'), last_price=('price', 'last'))

    # Вычисляем процентное изменение для каждой акции
    grouped['percentage_change'] = (grouped['last_price'] - grouped['first_price']) / grouped['first_price']

    # Находим топ акций по процентному изменению
    top_n_stocks = grouped.nlargest(n, 'percentage_change')
    print(top_n_stocks)

    unique_tics = pd.Series(np.concatenate((['RUB'], df['tic'].unique())))
    result = np.where(unique_tics.isin(top_n_stocks.index), 1. / n, 0.)
    return result


if __name__ == '__main__':
    _, trade = load_datasets()

    # profiler = LineProfiler()
    # profiler.add_function(top_stocks)
    # profiler_wrapper = profiler(top_stocks)
    modern_portfolio_theory(trade)
    print()
    black_litterman_theory(trade)
    # profiler.print_stats()
