import asyncio
import os

import pandas as pd
from tinkoff.invest import CandleInterval

from src.data_engine.fundamental import add_fundamental_indicator, load_fundamentals
from src.data_engine.loader import load_moex_data
from src.data_engine.preprocessor import preprocess_dataframe
from src.data_engine.technical import add_technical_indicator, add_return_lag_feature, add_turbulence_feature
from src.data_engine.utils import check_file_path
from src.data_engine.validator import StockQuoteValidator

if __name__ == '__main__':
    # токен для Invest API должен быть в этой переменной окружения
    # https://www.tbank.ru/invest/settings/api/
    token_str = os.environ['TOKEN']
    # даты должны быть в формате utils/DATE_FORMAT
    start_date = '2023-10-01 00:00:00'
    end_date = '2024-10-04 00:00:00'
    time_interval = CandleInterval.CANDLE_INTERVAL_DAY

    # название датасета
    # name = '2023-10-01_2024-10-04_HOUR'
    name = '{}_{}_{}'.format(start_date.split(' ')[0], end_date.split(' ')[0], str(time_interval)[31:])
    print(name)

    raw_path = f'../../data/raw/{name}.csv'
    check_file_path(raw_path)
    out_path = f'../../data/pre/{name}_final.csv'
    check_file_path(out_path)

    print('Выкачиваем котировки Мосбиржи...')
    result_df = asyncio.run(load_moex_data(token_str, start_date, end_date, time_interval))
    # чекпоинт. если при обработке данных возникнет ошибка, можно загрузиться отсюда и не выкачивать данные заново.
    result_df.to_csv(raw_path, index=False, encoding='utf-8')
    # result_df = pd.read_csv(raw_path)

    print('Заполняем пропуски, удаляем выбросы...')
    result_df = preprocess_dataframe(result_df)

    fin_data_path = '../../data/auxiliary/financial_data.csv'
    if not os.path.isfile(fin_data_path):
        print('Скачиваем финансовые данные Finam...')
        fin_data = load_fundamentals(result_df['tic'].unique())
    else:
        print('Загружаем фундаментальные индикаторы...')
        fin_data = pd.read_csv(fin_data_path)

    print('Добавляем технические индикаторы...')
    result_df = add_technical_indicator(result_df)
    print('Вычисляем индекс турбулентности...')
    result_df = add_turbulence_feature(result_df)
    print('Считаем процентное изменение цен...')
    result_df = add_return_lag_feature(result_df)
    print('Добавляем фундаментальные индикаторы...')
    result_df = add_fundamental_indicator(result_df, fin_data)

    print('Проверяем данные...')
    validator = StockQuoteValidator(threshold=0.8)
    result = validator.validate(result_df)
    for e in result:
        print(e)
        print()

    print('Сохраняем датасет')
    result_df.to_csv(out_path, index=False, encoding='utf-8')
