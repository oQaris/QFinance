import pandas as pd

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_start_end_dates(dataset):
    dates = pd.to_datetime(dataset['date'], format=DATE_FORMAT)

    start_date = dates.min().strftime(DATE_FORMAT)
    end_date = dates.max().strftime(DATE_FORMAT)

    return start_date, end_date


def split(dataset, train_ratio=0.8, stratification=0):
    dates = pd.to_datetime(dataset['date'], format=DATE_FORMAT)
    dataset['day'] = dates.dt.date

    unique_days = dataset['day'].unique()

    split_index = int(len(unique_days) * train_ratio)

    train_days = unique_days[:split_index]
    test_days = unique_days[split_index - stratification:]

    train_data = dataset[dataset['day'].isin(train_days)]
    test_data = dataset[dataset['day'].isin(test_days)]

    # Смещаем индексы так, чтобы они начинались с 0
    train_data.index = train_data.index - train_data.index.min()
    test_data.index = test_data.index - test_data.index.min()

    train = train_data.drop(columns=['day'])
    trade = test_data.drop(columns=['day'])

    return train, trade
