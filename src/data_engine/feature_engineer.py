import pandas as pd

from src.data_engine.fundamental import add_fundamental_indicator
from src.data_engine.technical import add_technical_indicator, add_return_lag_feature, add_turbulence_feature
from src.data_engine.utils import check_file_path

if __name__ == '__main__':
    name = '2024-01-01_2024-09-28_DAY'
    result_df = pd.read_csv(f'../../data/pre/{name}_preprocess.csv')
    fin_data = pd.read_csv(f'../../data/auxiliary/financial_data.csv')

    out_path = f'../../data/pre/{name}_final.csv'
    check_file_path(out_path)

    print('Добавляем технические индикаторы...')
    result_df = add_technical_indicator(result_df)
    print('Вычисляем индекс турбулентности...')
    result_df = add_turbulence_feature(result_df)
    print('Считаем процентное изменение цен...')
    result_df = add_return_lag_feature(result_df)
    print('Добавляем фундаментальные индикаторы...')
    result_df = add_fundamental_indicator(result_df, fin_data)

    print('Сохраняем датасет')
    result_df.to_csv(out_path, index=False, encoding='utf-8')
