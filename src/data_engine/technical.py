import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

INDICATORS = (
    "macd",  # Moving Average Convergence Divergence (стандартный)
    "macdh",  # MACD Histogram
    "rsi_14",  # Relative Strength Index с периодом 14
    "close_64_sma",  # Simple Moving Average (SMA) с периодом 64
    "close_64_ema",  # Exponential Moving Average (EMA) с периодом 64
    "boll",  # Средняя полоса Боллинджера
    "atr_14",  # Average True Range с периодом 14
    "adx",  # Average Directional Index с периодом 14
    "cci_14",  # Commodity Channel Index с периодом 14
    "stochrsi",  # Стохастический RSI
    "wr_14",  # Williams %R с периодом 14
    "pdi",  # +DI индикатор
    "ndi",  # -DI индикатор
    "trix",  # Trix индикатор с периодом 9
    "dma",  # Demand Index
    "cmo",  # Моментум с периодом 14
    "close_14_roc"  # Rate of Change (Темп изменения)
)


def add_technical_indicator(data, indicator_list=INDICATORS):
    """
    Вычислить технические индикаторы, используя stockstats
    """
    df = data.copy()
    df = df.sort_values(by=["tic", "date"])
    stock = Sdf.retype(df.copy())
    unique_ticker = stock.tic.unique()

    for indicator in indicator_list:
        print(f'process {indicator}')
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["tic"] = unique_ticker[i]
                temp_indicator["date"] = df[df.tic == unique_ticker[i]]["date"].to_list()
                indicator_df = pd.concat([indicator_df, temp_indicator], axis=0, ignore_index=True)
            except Exception as e:
                print(e)
        indicator_df = indicator_df.ffill().bfill()
        df = df.merge(indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left")
    df = df.sort_values(by=["date", "tic"])
    return df


def add_return_lag_feature(data):
    """
    Рассчитать процентное изменение цен в разных диапазонах
    """
    df = data.copy()
    lags = [1, 7, 29]
    dfs = []
    for tic in df['tic'].unique():
        df_tic = df[df['tic'] == tic].copy()
        for lag in lags:
            lag_col = f"return_lag_{lag}"
            # Считаем процентное изменение 100*(d[i]-d[i-lag])/d[i-lag]
            df_tic[lag_col] = df_tic['close'].pct_change(lag)
        df_tic = df_tic.fillna(0)
        dfs.append(df_tic)
    # Объединяем все таблицы в одну
    df = pd.concat(dfs)
    return df


def add_turbulence_feature(data):
    """
    Рассчитать индекс турбулентности рынка
    """
    # can add other market assets
    df = data.copy()
    unique_date = df['date'].unique()
    # start after window size
    window = max(int(len(unique_date) * 0.13), 79)
    df_price_pivot = df.pivot(index='date', columns='tic', values='close')
    # use returns to calculate turbulence
    df_price_pivot = df_price_pivot.pct_change()

    turbulence_index = [0] * window
    count = 0
    for i in range(window, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        # use one year rolling window to calcualte covariance
        hist_price = df_price_pivot[
            (df_price_pivot.index < unique_date[i])
            & (df_price_pivot.index >= unique_date[i - window])]
        # Drop tickers which has number missing values more than the "oldest" ticker
        filtered_hist_price = hist_price.iloc[
                              hist_price.isna().sum().min():
                              ].dropna(axis=1)

        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
            filtered_hist_price, axis=0)

        temp = (current_temp.values.dot(np.linalg.pinv(cov_temp))
                .dot(current_temp.values.T))
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)
    try:
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index})
    except ValueError:
        raise Exception("Turbulence information could not be added.")

    df = df.merge(turbulence_index, on="date")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    return df


def add_time_features(data):
    """
    Выделить из даты компоненты и преобразовать временные метки в периодические признаки
    """
    df = data.copy()
    date = pd.to_datetime(df['date'])

    month = date.dt.month
    day = date.dt.day
    day_of_week = date.dt.dayofweek
    hour = date.dt.hour
    minute = date.dt.minute

    # Применение тригонометрических преобразований
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    df['day_sin'] = np.sin(2 * np.pi * day / 31)  # 31 день в максимальном месяце
    df['day_cos'] = np.cos(2 * np.pi * day / 31)

    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * minute / 60)

    # Удаление столбцов, значения в которых одинаковые (если дата содержит только дни)
    cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=cols_to_drop, inplace=True)
    return df
