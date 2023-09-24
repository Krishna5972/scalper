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

timeframe = '15m'


while True:
    current_minute = datetime.now().minute
    
    if current_minute not in [0, 15, 30, 45]:
        a = datetime.utcnow()
        data= get_scaner_data(sleep_time=3600)
        possible_long,possible_short,possible_volatile = get_coins(data)
        print(f'Started to check  : {a}')
        
        for coin in possible_long:
            try:
                if is_long_tradable(coin,timeframe):
                    print(f'{coin} can be longed')
            except Exception as e:
                print(f'{coin} error',e)

        print('Done checking for longs')
        
        print('Started to check for shorts')
        
        for coin in possible_short:
            try:
                if is_short_tradable(coin, timeframe):
                    print(f'{coin} can be shorted')
            except Exception as e:
                print(f'{coin} error',e)
        print('Done checking for shorts')

        print('Started to check for volatile')
        for coin in possible_volatile:
            try:
                if is_volatile_tradable(coin, timeframe):
                    print(f'{coin} can be traded')
            except Exception as e:
                print(f'{coin} error',e)
        b = datetime.utcnow()
        print(f'Done checking : {b}')
        elapsed_time = b - a
        print(f"Time taken: {elapsed_time.total_seconds()} seconds")

        print('==========================================')
        break
