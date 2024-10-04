import asyncio
import datetime
import decimal
import os

import pandas as pd
from tinkoff.invest import AsyncClient, CandleInterval, InstrumentShort
from tinkoff.invest import Instrument
from tinkoff.invest import Quotation
from tinkoff.invest import RealExchange
from tinkoff.invest.async_services import AsyncServices
from tqdm import tqdm

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# https://tinkoff.github.io/investAPI/faq_custom_types/

def convert_to_quotation(value):
    value = decimal.Decimal(value)
    quotation = Quotation(
        units=int(value) if value is not None else 0,
        nano=int((value % 1) * decimal.Decimal(1000000000)) if value is not None else 0
    )
    return quotation


def convert_to_double(quotation):
    if quotation.units == 0 and quotation.nano == 0:
        return 0.0
    else:
        return float(quotation.units) + (float(quotation.nano) / 1000000000)


def string_to_utc_datetime(str):
    return datetime.datetime.strptime(str, DATE_FORMAT).replace(tzinfo=datetime.timezone.utc)


def datetime_to_utc_string(dt):
    return dt.strftime(DATE_FORMAT)


async def load_candles_instr(client: AsyncServices, instr: Instrument, start: str, end: str, interval: CandleInterval):
    candles = None
    async for candle in client.get_all_candles(
            figi=instr.figi,
            from_=string_to_utc_datetime(start),
            to=string_to_utc_datetime(end),
            interval=interval,
    ):
        lot = 1 if isinstance(instr, InstrumentShort) or instr.lot is None else instr.lot
        next_line = {
            "tic": instr.ticker,
            "date": datetime_to_utc_string(candle.time),
            "open": convert_to_double(candle.open),
            "close": convert_to_double(candle.close),
            "high": convert_to_double(candle.high),
            "low": convert_to_double(candle.low),
            "volume": candle.volume,
            "lot": lot
        }
        candles = pd.concat([candles, pd.DataFrame.from_records([next_line])])
    return candles


async def get_vix(client: AsyncServices):
    all_vix = await client.instruments.find_instrument(query="VIX")
    all_vix = all_vix.instruments
    vix = [i for i in all_vix if i.instrument_type == 'index']
    if len(vix) > 1:
        raise Exception("VIX mast be unique")
    return vix[0]


async def load_moex_data(token, start: str, end: str, interval: CandleInterval):
    acc = None
    async with AsyncClient(token) as client:
        all_shares = await client.instruments.shares()
        moex_shares = [i for i in all_shares.instruments
                       if i.api_trade_available_flag == True
                       and i.real_exchange == RealExchange.REAL_EXCHANGE_MOEX]
        vix = await get_vix(client)
        need_shares = [vix] + moex_shares

        print("total instruments:", len(need_shares))
        for instr in (pbar := tqdm(need_shares)):
            pbar.set_description(f"Processing {instr.ticker}")
            acc = pd.concat([acc, await load_candles_instr(client, instr, start, end, interval)])

    return acc


if __name__ == '__main__':
    # должно быть в формате DATE_FORMAT
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-09-28 19:00:00'
    time_interval = CandleInterval.CANDLE_INTERVAL_DAY
    # токен для Invest API должен быть в этой переменной окружения
    # https://www.tbank.ru/invest/settings/api/
    token_str = os.environ['TOKEN']

    result_df = asyncio.run(load_moex_data(token_str, start_date, end_date, time_interval))
    result_df.to_csv("data/24_01-10_moex-vix-4hour.csv", index=False, sep=',', encoding='utf-8')
