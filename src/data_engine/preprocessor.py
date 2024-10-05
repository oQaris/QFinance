def fill_missing_dates_simple(group, dates):
    return group.set_index('date') \
        .reindex(dates) \
        .reset_index() \
        .ffill().bfill()


def fill_missing_dates_expert(group, dates):
    # Создаем исходный индекс для проверки на новые строки после reindex
    original_index = group.set_index('date').index

    # Заполняем пропущенные даты
    filled_group = group.set_index('date') \
        .reindex(dates) \
        .reset_index()

    # Определяем, какие строки были добавлены
    new_rows = ~filled_group['date'].isin(original_index)

    # Пропущенные значения в 'close' заполняем сначала вперёд, потом назад
    filled_group['close'] = filled_group['close'].ffill().bfill()

    # значения свечи ставим равными цене закрытия, а объём нулевой
    filled_group.loc[new_rows, 'open'] = filled_group.loc[new_rows, 'close']
    filled_group.loc[new_rows, 'high'] = filled_group.loc[new_rows, 'close']
    filled_group.loc[new_rows, 'low'] = filled_group.loc[new_rows, 'close']
    filled_group.loc[new_rows, 'volume'] = 0

    # Заполняем остальные столбцы
    return filled_group.ffill().bfill()


def preprocess_dataframe(df_input):
    df = df_input.copy()

    # Транспонирование VIX

    vix_data = df[df['tic'] == 'VIX'][['date', 'close']].rename(columns={'close': 'vix'})
    if len(vix_data) == 0:
        raise ValueError('В датасете отсутствует VIX')
    df = df.merge(vix_data, on='date', how='left')
    df = df[df['tic'] != 'VIX']
    print('VIX транспонирован, остались столбцы:\n', df.columns.tolist())

    # Удаление тикеров с недостатком данных

    tic_counts = df['tic'].value_counts()
    print(tic_counts)
    print()

    median_count = tic_counts.median()
    threshold = median_count / 1.5
    len_old = len(df)
    unfulfilled = tic_counts[tic_counts < threshold].index
    df = df[~df['tic'].isin(unfulfilled)]
    print(f'Удалено {len(unfulfilled)} тикеров ({len_old - len(df)} строк):', unfulfilled.tolist())

    # Удаление дубликатов

    duplicates = df[df.duplicated(subset=['tic', 'date'], keep=False)]
    if not duplicates.empty:
        unique_values = duplicates.groupby(['tic', 'date']) \
            .apply(lambda x: x.nunique())

        if (unique_values > 1).any().any():
            raise Exception('Дубликаты date для одного tic имеют разные значения в различных столбцах')

    len_old = len(df)
    df = df.drop_duplicates(subset=['tic', 'date'])
    print(f'Удалено {len_old - len(df)} дубликатов\n')

    # Заполнение пропусков по датам

    all_dates = df['date'].unique()
    all_dates.sort()
    print(f'FROM {all_dates.min()}\nTO   {all_dates.max()}')

    len_old = len(df)
    df = df.groupby('tic') \
        .apply(fill_missing_dates_expert, all_dates, include_groups=False) \
        .reset_index() \
        .drop(columns='level_1')
    print(f'Заполнено {len(df) - len_old} пропусков дат')

    print()
    print(df['tic'].value_counts())
    print()
    print(df.head())
    return df
