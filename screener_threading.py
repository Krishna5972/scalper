import threading
import logging
import pandas as pd
import time
from binance.client import Client
import json
import requests
from datetime import datetime
logging.basicConfig(filename='trading_data_log.txt',  filemode='a',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import datetime,timedelta
from data_extraction import *
import config

from functions import *

timeframes =['15m','30m','1h']

def check_longs(possible_long,timeframe):
    for coin in possible_long:
        try:
            if is_long_tradable(coin, timeframe):
                print(f'{coin} can be longed')
        except Exception as e:
            print(f'{coin} error', e)
    print('Done checking for longs')

def check_shorts(possible_short,timeframe):
    for coin in possible_short:
            try:
                if is_short_tradable(coin, timeframe):
                    print(f'{coin} can be shorted')
            except Exception as e:
                print(f'{coin} error',e)
    print('Done checking for shorts')

def check_volatile(possible_volatile,timeframe):
    for coin in possible_volatile:
            try:
                if is_volatile_tradable(coin, timeframe):
                    print(f'{coin} can be traded')
            except Exception as e:
                print(f'{coin} error',e)
    print('Done checking for Volatile')



timeframes_map = {
    0: ['15m', '30m', '1h'],
    15: ['15m'],
    30: ['15m', '30m'],
    45: ['15m']
}

while True:
    current_minute = datetime.now().minute
    if current_minute not in [0, 15, 30, 45]:
        timeframes = timeframes_map.get(current_minute, ['15m', '30m', '1h'])

        a = datetime.utcnow()
        data = get_scaner_data(sleep_time=3600)
        possible_long, possible_short, possible_volatile = get_coins(data)
        print(f'Started to check  : {a}')

        for timeframe in timeframes:
            print(f'===========================timeframe : {timeframe}========================================')
            threads = [
                threading.Thread(target=check_longs, args=(possible_long, timeframe)),
                threading.Thread(target=check_shorts, args=(possible_short, timeframe)),
                threading.Thread(target=check_volatile, args=(possible_volatile, timeframe))
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            b = datetime.utcnow()
            print(f'Done checking : {b}')
            elapsed_time = b - a
        print(f"Time taken: {elapsed_time.total_seconds()} seconds")
    break