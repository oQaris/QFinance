import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

from src.data_engine.utils import check_file_path, string_to_utc_datetime


def fetch_finam_data(driver, ticker: str):
    """
    Выкачать и распарсить фундаментальные индикаторы из сервиса Finam для конкретного тикера
    """
    url = f'https://www.finam.ru/quote/moex/{ticker}/financial/'
    driver.get(url)
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    # Ищем таблицу "Мультипликаторы"
    multipliers_table = soup.find('table', class_='finfin-local-plugin-quote-item-financial-table')

    # Если таблица не найдена, возвращаем пустой DataFrame
    if multipliers_table is None:
        return pd.DataFrame()

    # Заголовки столбцов (года)
    headers = [th.get_text(strip=True) for th in multipliers_table.find('thead').find_all('th')[1:]]

    # Получаем строки данных
    data = []
    for row in multipliers_table.find('tbody').find_all('tr'):
        cols = row.find_all('td')

        # Получаем название показателя, игнорируя содержимое <span>
        # todo найти публичный аналог _all_strings()
        indicator_name = next(cols[0].find('div', class_='p05x')._all_strings(True))

        # Значения по годам
        values = [col.get_text(strip=True) or None for col in cols[1:]]
        data.append([indicator_name] + values)

    # Создание DataFrame для удобства отображения
    df = pd.DataFrame(data, columns=['indicator'] + headers)
    df['tic'] = ticker
    return df


def load_fundamentals(tics):
    """
    Получить фундаментальные данные для всех переданных тикеров, произвести очистку
    """
    chrome_driver = webdriver.Chrome()

    fundamentals = pd.DataFrame()
    for tic in tics:
        ticker_data = fetch_finam_data(chrome_driver, tic)
        print(ticker_data)
        fundamentals = pd.concat([fundamentals, ticker_data], axis=0)
    chrome_driver.close()

    # Перестановка столбцов: сначала Ticker, затем Indicator, потом года
    fundamentals.reset_index(drop=True, inplace=True)
    columns_order = ['tic', 'indicator'] + sorted(
        [col for col in fundamentals.columns if col not in ['tic', 'indicator']])
    fundamentals = fundamentals[columns_order]

    # 1. Удаление индикаторов, по которым мало данных
    # todo считать медиану
    fundamentals = fundamentals[fundamentals['indicator'] != 'Валовая маржа']
    # 2. Заменяем запятые на точки в числовых значениях
    fundamentals = fundamentals.replace(',', '.', regex=True)
    # 3. Заменяем пустые значения на NaN
    fundamentals = fundamentals.replace('', np.nan)
    # 4. Удаление строк, где для всех годов нет данных
    year_columns = [col for col in fundamentals.columns if col.isdigit()]
    fundamentals = fundamentals.dropna(subset=year_columns, how='all')
    # 5. Удаление столбцов, в которых все значения NaN
    fundamentals = fundamentals.dropna(axis=1, how='all')
    return fundamentals


def add_fundamental_indicator(data, fundamentals):
    final_data = data.copy()
    # Добавляем колонку с годом для каждой строки
    final_data['year'] = final_data['date'].apply(lambda x: string_to_utc_datetime(x).year)
    grouped = final_data.groupby(['tic', 'year'])

    # Оптимизированная функция для применения модификаторов на уровне группы
    def apply_modifiers(group, fnd):
        ticker = group['tic'].iloc[0]
        year = group['year'].iloc[0]
        # Индикаторы для тикера из группы
        fnd_for_ticker = fnd[fnd['tic'] == ticker]

        for ind_name in fnd_for_ticker['indicator'].unique():
            available_years = [col for col in fnd_for_ticker.columns if col.isdigit()]
            # Перебираем года в порядке X, X-1, X+1, X-2, X+2, и т.д.
            available_years = sorted(available_years, key=lambda x: abs(int(x) - year))
            for y in available_years:
                # Найти значение индикатора для ближайшего доступного года
                ind_value = fnd_for_ticker[(fnd_for_ticker['indicator'] == ind_name)][y]
                if not ind_value.isna().all():
                    group[ind_name] = ind_value.values[0]
                    break
            else:  # Если ничего не найдено
                group[ind_name] = 0
        return group

    # Применяем функцию ко всем группам
    final_data = grouped.apply(lambda group: apply_modifiers(group, fundamentals))
    # Удаляем временную колонку с годом
    final_data = final_data.reset_index(drop=True).drop(columns=['year'])
    final_data = final_data.fillna(0)
    return final_data


if __name__ == '__main__':
    path = f'../../data/auxiliary/financial_data.csv'
    check_file_path(path)

    moex_tics = pd.read_csv('../../data/auxiliary/moex_tics.csv')
    # Раскомментировать для загрузки и парсинга
    # fin_data = load_fundamentals(moex_tics['tic'])
    # fin_data.to_csv(path, index=False)
    # Используем уже обработанные данные
    fin_data = pd.read_csv(path)

    # Небольшой анализ индикаторов
    print('\nНе удалось извлечь данные для:')
    print(moex_tics[~moex_tics.isin(fin_data['tic'].unique())].dropna())

    print('\nПроцент наполненности тикеров:')
    unique_indicators = fin_data['indicator'].nunique()
    tic_counts = fin_data.groupby('tic')['indicator'].nunique()
    tic_presence = (tic_counts / unique_indicators) * 100
    tic_presence = tic_presence.sort_values(ascending=False)
    print(tic_presence)

    print('\nПроцент наполненности индикаторов:')
    fin_data = pd.read_csv(path)
    unique_tickers = fin_data['tic'].nunique()
    indicator_counts = fin_data.groupby('indicator')['tic'].nunique()
    indicator_presence = (indicator_counts / unique_tickers) * 100
    indicator_presence = indicator_presence.sort_values(ascending=False)
    print(indicator_presence)
