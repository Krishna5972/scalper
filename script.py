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

timeframe ='3m'
period = 14
atr1 = 1.2


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

stream = f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_3m"

df = df[['OpenTime', 'open', 'high', 'low', 'close', 'volume']]


def get_signal(super_df):
    signal = ['Buy' if super_df.iloc[-1]
                                    ['in_uptrend'] == True else 'Sell'][0]
    
    return signal

def get_entry(super_df):
    return super_df.iloc[-1]['close']

def get_stake(super_df,client,risk = 0.02):
    signal = get_signal(super_df)
    entry = get_entry(super_df)
    client._create_futures_api_uri = create_futures_api_uri_v2.__get__(client, Client)
    acc_balance = round(float(client.futures_account()['totalCrossWalletBalance']), 2)
    client._create_futures_api_uri = create_futures_api_uri_v1.__get__(client, Client)
    stake = (acc_balance*0.88)
    notifier(f'USDT : Allocated stake:{round(stake,2)}')
    if signal == 'Buy':
        sl = super_df.iloc[-1]['lowerband']
        sl_perc = (entry-sl)/entry
    else:
        sl = super_df.iloc[-1]['upperband']
        sl_perc = (sl-entry)/entry

    stake = (stake*risk)/sl_perc
    
    return stake

def get_over_all_trend(coin):
    str_date = (datetime.now()- timedelta(days=20)).strftime('%b %d,%Y')
    end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')

    df=dataextract(coin,str_date,end_str,'1d',client)
    df['Date'] = pd.to_datetime(df['OpenTime'])
    df.set_index('Date', inplace=True)

    df['ROC_6'] = df['close'].pct_change(periods=6) * 100
    df['ROC_EMA'] = df['ROC_6'].ewm(span=3, adjust=False).mean()

    
    latest_ROC_EMA = df['ROC_EMA'].iloc[-1]
    if latest_ROC_EMA > 0:
        trend = "Uptrend"
    elif latest_ROC_EMA < 0:
        trend = "Downtrend"
    else:
        trend = "Neutral"

    print(f"Latest Rate of Change EMA: {latest_ROC_EMA:.2f}%")
    print(f"Overall trend: {trend}")
    
    return trend

def get_latest_df(data,df):
    candle = data['k']
    candle_data = [candle['t'], candle['o'],
                candle['h'], candle['l'], candle['c'], candle['v']]
    temp_df = pd.DataFrame([candle_data], columns=[
                        'OpenTime', 'open', 'high', 'low', 'close', 'volume'])
    temp_df['OpenTime'] = temp_df['OpenTime'] / 1000  
    temp_df['OpenTime'] = datetime.fromtimestamp(temp_df['OpenTime'])
    df = pd.concat([df, temp_df])
    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols:
        df[col] = df[col].astype(float)
    
    
    df.reset_index(drop=True,inplace=True)
    return df

def close_any_open_positions():
    try:
        # close open position if any
        close_position(client, coin, 'Sell')
        in_trade_usdt.value = 0
        notifier(f'USDT : Position Closed {timeframe}')
    except Exception as err:
        try:
            close_position(client, coin, 'Buy')
            notifier(f'USDT : Position Closed {timeframe}')
            in_trade_usdt.value = 0
        except Exception as e:
            notifier(f'USDT : No Open Position to Close {timeframe}')

def close_position(client, coin, signal):
    if signal == 'Buy':
        client.futures_create_order(
            symbol=f'{coin}USDT', side='SELL', type='MARKET', quantity=1000, dualSidePosition=True, positionSide='LONG')
    else:
        client.futures_create_order(
            symbol=f'{coin}USDT', side='BUY', type='MARKET', quantity=1000, dualSidePosition=True, positionSide='SHORT')
        
def get_pivot_supertrend_signal(pivot_super_df):
    trend= pivot_super_df.iloc[-1]['in_uptrend']
    print(type(trend))
    if trend == True:
        return "Buy"
    else:
        return "Sell"
    
async def on_message(message,df):
    data = json.loads(message)
    if data['k']['x'] == True:
        notifier(f'Candle closed : {timeframe} , coin : {coin}')
        df = get_latest_df(data, df)
        df_copy = df.copy()
        super_df=supertrend_njit(coin, df_copy, period, atr1)
        pivot_st = PivotSuperTrendConfiguration()
        print(f'Length of super_df : {super_df.shape[0]}')
        df_copy = df.copy()
        pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
        print(super_df[['OpenTime','close','upperband', 'lowerband','in_uptrend']].iloc[-5:])
        print('======================================== Pivot point below =========================')
        print(pivot_super_df[['OpenTime','close','upper_band', 'lower_band','in_uptrend']].iloc[-5:])
        
        if (super_df.iloc[-1]['in_uptrend'] != super_df.iloc[-2]['in_uptrend']) or (pivot_super_df.iloc[-1]['in_uptrend'] != pivot_super_df.iloc[-2]['in_uptrend']):          
            close_any_open_positions()
            signal = get_signal(super_df)
            entry =  get_entry(super_df)              
            over_all_trend = get_over_all_trend(coin)
      
            
        
            tradeConfig = TradeConfiguration()
            risk = tradeConfig.get_risk(over_all_trend,signal)
            
            stake = get_stake(super_df,client,risk)
            quantity = round(stake/entry, round_quantity)
            
            
            
            order = Order(coin = coin,
                          entry = entry,
                          quantity = quantity,
                          round_price = round_price 
                         )
            
            
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            
            
            
            if signal == 'Buy' and pivot_signal == 'Buy':
                order.make_buy_trade(client)
            elif signal == 'Sell' and pivot_signal == 'Sell':
                order.make_sell_trade(client)
            else:
                notifier(f'Short Term Trend : {signal} , Pivot SuperTrend signal : {pivot_signal} , not taking the trade')
            
            notifier(f'Overall trend before passing : {over_all_trend}')
            notifier(f'signal : {signal},entry : {entry},stake : {stake},quantity : {quantity}')
            notifier(f'Short Term Trend : {signal} , Pivot SuperTrend signal : {pivot_signal}')
        notifier(f'Pivot trend : {pivot_super_df["in_uptrend"].iloc[-1]} , supertrend : {super_df["in_uptrend"].iloc[-1]}')
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

