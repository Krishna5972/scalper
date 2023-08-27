import requests
import pandas as pd
from datetime import datetime
import time
import ccxt
import config
from binance.client import Client
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import multiprocessing
from datetime import datetime,timedelta
from data_extraction import *
import websocket
import json
import asyncio
import nest_asyncio
nest_asyncio.apply()
from modules import *


from functions import *

telegram_auth_token='5515290544:AAG9T15VaY6BIxX2VYX8x2qr34aC-zVEYMo'
telegram_group_id='notifier2_scanner_bot_link'


        
exchange = ccxt.binance({
    "apiKey": config.api_key,
    "secret": config.secret_key,
    'options': {
    'defaultType': 'future',
    },
})

coin = input("Please enter the coin name: ")

print(f"You entered: {coin}")

stream = f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_{timeframe}"


usdt_leverage,busd_leverage = 25,25

max_usdt_leverage,max_busd_leverage = get_max_leverage(coin, config.api_key, config.secret_key)

usdt_leverage = min(usdt_leverage, max_usdt_leverage)
busd_leverage = min(busd_leverage, max_busd_leverage)

is_usdt_exist=1
is_busd_exist=1


while(True):
    try:
        client=Client(config.api_key,config.secret_key)
        try:
            client.futures_change_leverage(symbol=f'{coin}USDT', leverage=usdt_leverage)
        except Exception as e:
            try:
                client.futures_change_leverage(symbol=f'{coin}USDT', leverage=max_usdt_leverage)
                notifier(f"Had to make a leverage change from {usdt_leverage} to {max_usdt_leverage}")
            except Exception as e:
                notifier(f'{coin}USDT Symbol does not Exist')
                is_usdt_exist=0
        try:
            client.futures_change_leverage(symbol=f'{coin}BUSD', leverage=busd_leverage)
        except Exception as e:
            try:
                client.futures_change_leverage(symbol=f'{coin}BUSD', leverage=max_busd_leverage)
            
                notifier(f"Had to make a leverage change from {busd_leverage} to {max_busd_leverage}")
            except Exception as e:
                notifier(f'{coin}BUSD Symbol does not Exist')
                is_busd_exist=0
            
        notifier(f'SARAVANA BHAVA')
        break
    except Exception as e:
        notifier(f'Met with exception {e}, sleeping for 5 minutes and trying again')
        time.sleep(300)


client._create_futures_api_uri = create_futures_api_uri_v2.__get__(client, Client)
in_trade_usdt,in_trade_busd = 0,0
if is_usdt_exist==1:
    pos=client.futures_position_information(symbol=f'{coin}USDT')
    if float(pos[0]['positionAmt']) !=0 or float(pos[1]['positionAmt']) !=0:
        in_trade_usdt=1

if is_busd_exist == 1:
    pos=client.futures_position_information(symbol=f'{coin}BUSD')
    if float(pos[0]['positionAmt']) !=0 or float(pos[1]['positionAmt']) !=0:
        in_trade_busd=1
client._create_futures_api_uri = create_futures_api_uri_v1.__get__(client, Client)



str_date = (datetime.now()- timedelta(days=5)).strftime('%b %d,%Y')
end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')

df=dataextract(coin,str_date,end_str,timeframe,client)



x_str = str(df['close'].iloc[-1])
decimal_index = x_str.find('.')
round_price = len(x_str) - decimal_index - 1



exchange_info = client.futures_exchange_info()

for symbol in exchange_info['symbols']:
    if symbol['symbol'] == f"{coin}USDT":
        round_quantity = symbol['quantityPrecision']
        break
df_copy = df.copy()

super_df=supertrend_njit(coin, df_copy, period, atr1)
df_copy = df.copy()
trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)

import asyncio
import json
import websockets



df = df[['OpenTime', 'open', 'high', 'low', 'close', 'volume']]



    


    
async def on_message(message,df):
    data = json.loads(message)

    if data['k']['x'] == True:
        notifier(f'Candle closed : {timeframe} , coin : {coin}')
        df = get_latest_df(data, df)
        df_copy = df.copy()
        super_df=supertrend_njit(coin, df_copy, period, atr1)
        ema = get_ema(super_df,'ema_81')
        
        print(f'Length of super_df : {super_df.shape[0]}')
        df_copy = df.copy()
        signal = get_signal(super_df)
        current_signal = signal
        prev_signal = get_prev_signal(super_df)

        pivot_st = PivotSuperTrendConfiguration()
        pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
        pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
        current_pivot_signal = pivot_signal
        prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

        print(f'Prev PivotSuperTrend signal : {prev_pivot_signal},Prev SuperTrend Signal : {prev_signal}' )

        print(f'Current PivotSuperTrend signal : {current_pivot_signal}, SuperTrend Signal : {current_signal}' )
        
        if (current_signal != prev_signal) or (current_pivot_signal !=prev_pivot_signal): 
            
            close_any_open_positions(coin,client)
            cancel_all_open_orders(coin,client)
            
            entry =  get_entry(super_df)              
            over_all_trend = get_over_all_trend(coin)
            lowerband = get_lowerband(super_df)
            upperband = get_upperband(super_df)

            ema = get_ema(super_df,'ema_81')
            
            tradeConfig = TradeConfiguration()
            risk = tradeConfig.get_risk(over_all_trend,signal)
            
            stake = get_stake(super_df,client,risk)
            stake = 60
            quantity = round(stake/entry, round_quantity)
            partial_profit_take = round(quantity/2,round_quantity)
            
            change = None
            
            if (current_pivot_signal !=prev_pivot_signal): 
                change = 'longTerm'
            
            if (current_signal != prev_signal):
                change = 'shortTerm'
            
            order = Order(coin = coin,
                          entry = entry,
                          quantity = quantity,
                          round_price = round_price,
                          change = change,
                          partial_profit_take = partial_profit_take,
                          lowerband = lowerband,
                          upperband = upperband
                         )
            
            if pivot_signal == 'Buy' and signal == 'Buy':  
                order.make_buy_trade(client)   
                notifier('No stoploss but take profit')
                
            elif pivot_signal == 'Buy' and signal == 'Sell' and change == 'shortTerm' and entry > ema:
                order.make_inverse_buy_trade(client)
                notifier(f'Made a inverse trade should have stop loss and a take profit')
            
            elif pivot_signal == 'Sell' and signal == 'Sell':
                order.make_sell_trade(client)
                notifier('No stoploss but take profit')
            elif pivot_signal == 'Sell' and signal == 'Buy' and change == 'shortTerm' and entry < ema:
                order.make_inverse_sell_trade(client)
                notifier(f'Made a inverse trade should have stop loss and take profit')
                
            else:
                notifier(f'Something is wrong...Debug')
            
#             notifier(f'Overall trend before passing : {over_all_trend}')
#             notifier(f'signal : {signal},entry : {entry},stake : {stake},quantity : {quantity}')
            notifier(f'Short Term Trend : {signal} , Pivot SuperTrend signal : {pivot_signal}')
    return df

async def listen(df):
    async with websockets.connect(stream) as ws:
        try:
            while True:
                message = await ws.recv()
                df = await on_message(message,df)
                
        except websockets.ConnectionClosed:
            print("WebSocket connection closed. Attempting to reconnect...")
            await asyncio.sleep(10)
            df = await listen(df)
    return df

async def main(df):
    while True:
#         try:
        df = await listen(df)
#         except Exception as e:
#             print(f"Error: {e}. Retrying in 10 seconds...")
#             await asyncio.sleep(10)



# Run the asyncio event loop
asyncio.run(main(df))

