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
from multiprocessing import Process, Manager

from functions import *

timeframes =['15m','30m','1h']

def check_longs(possible_long,timeframe,final_list):
    for coin in possible_long:
        try:
            if is_long_tradable(coin, timeframe):
                value = is_long_tradable(coin, timeframe)
                x = {
                         'coin' : coin,
                         'stake' : 0.5,
                         'timeframe' : timeframe,
                         'strategy' : 'long',
                         'time' : datetime.utcnow()

                     }
                if value == 1:
                    x['stake'] = 0.5     
                else:
                    x['stake'] = 1
                print(f'{coin} can be longed')

                final_list.append(x)

        except Exception as e:
            print(f'{coin} error', e)
    print('Done checking for longs')

def check_shorts(possible_short,timeframe,final_list):
    for coin in possible_short:
            try:
                if is_short_tradable(coin, timeframe):
                    value = is_short_tradable(coin, timeframe)
                    x = {
                            'coin' : coin,
                            'stake' : 0.5,
                            'timeframe' : timeframe,
                            'strategy' : 'short',
                            'time' : datetime.utcnow()
                        }
                    if value == 1:
                        x['stake'] = 0.5     
                    else:
                        x['stake'] = 1
                    

                    final_list.append(x)
                    print(f'{coin} can be shorted')
            except Exception as e:
                print(f'{coin} error',e)
    print('Done checking for shorts')

def check_volatile(possible_volatile,timeframe,final_list):
    for coin in possible_volatile:
            try:

                is_tradable,signal = is_volatile_tradable(coin, timeframe)
                if is_tradable:
                    x = {
                            'coin' : coin,
                            'stake' : 1,
                            'timeframe' : timeframe,
                            'strategy' : 'volatile',
                            'signal' : signal,
                            'time' : datetime.utcnow()
                        }
                    final_list.append(x)
                    
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

def worker(possible_long, possible_short, possible_volatile, timeframe, final_list):
    threads = [
        Process(target=check_longs, args=(possible_long, timeframe, final_list)),
        Process(target=check_shorts, args=(possible_short, timeframe)),
        Process(target=check_volatile, args=(possible_volatile, timeframe))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()



if __name__ == "__main__":
    while True:
        with Manager() as manager:
            final_list = manager.list()
            current_minute = datetime.now().minute
            if current_minute not in [0, 15, 30, 45]:
                timeframes = timeframes_map.get(current_minute, ['15m', '30m', '1h'])

                a = datetime.utcnow()
                data = get_scaner_data(sleep_time=3600)
                possible_long, possible_short, possible_volatile = get_coins(data)

                possible_long = [item for item in possible_long if item not in possible_volatile]
                possible_short = [item for item in possible_short if item not in possible_volatile]

                print(f'Number of longs : {len(possible_long)}')
                print(f'Number of shorts : {len(possible_short)}')
                print(f'Number of volatile coins : {len(possible_volatile)}')
                

                # Splitting lists into two
                half_size_long = len(possible_long) // 2
                half_size_short = len(possible_short) // 2
                half_size_volatile = len(possible_volatile) // 2

                long1, long2 = possible_long[:half_size_long], possible_long[half_size_long:]
                short1, short2 = possible_short[:half_size_short], possible_short[half_size_short:]
                volatile1, volatile2 = possible_volatile[:half_size_volatile], possible_volatile[half_size_volatile:]

                print(f'Started to check : {a}')
                
                for timeframe in timeframes:
                    print(f'===========================timeframe : {timeframe}========================================')

                    if timeframe == '15m':
                        processes = [
                        Process(target=check_volatile, args=(volatile1, timeframe, final_list)),
                        Process(target=check_volatile, args=(volatile2, timeframe, final_list))
                    ]
                    else:
                        processes = [
                        Process(target=check_longs, args=(long1, timeframe, final_list)),
                        Process(target=check_longs, args=(long2, timeframe, final_list)),
                        Process(target=check_shorts, args=(short1, timeframe, final_list)),
                        Process(target=check_shorts, args=(short2, timeframe, final_list)),
                        Process(target=check_volatile, args=(volatile1, timeframe, final_list)),
                        Process(target=check_volatile, args=(volatile2, timeframe, final_list))
                    ]

                    for process in processes:
                        process.start()

                    for process in processes:
                        process.join()

                    b = datetime.utcnow()
                    print(f'Done checking : {b}')
                    elapsed_time = b - a
                print(f"Time taken: {elapsed_time.total_seconds()} seconds")

                final_list = list(final_list)
            

                break

    timeframes = ['15m', '30m', '1h']
    strategies = ['volatile', 'long', 'short']

    result = {timeframe: {strategy: [] for strategy in strategies} for timeframe in timeframes}

    # Populating the structure
    for entry in final_list:
        timeframe = entry['timeframe']
        strategy = entry['strategy']

        coin_info = {
            'coin': entry['coin'],
            'stake': entry['stake'],
            'time': entry['time']
        }

        if strategy == 'volatile':
            coin_info['signal'] = entry['signal']

        result.setdefault(timeframe, {}).setdefault(strategy, []).append(coin_info)
                

    print(result)