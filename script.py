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
from modules import *
from multiprocessing import Process, Manager


from functions import *


async def main(shared_coin,current_trade):
    telegram_auth_token='5515290544:AAG9T15VaY6BIxX2VYX8x2qr34aC-zVEYMo'
    telegram_group_id='notifier2_scanner_bot_link'


    
            
    exchange = ccxt.binance({
        "apiKey": config.api_key,
        "secret": config.secret_key,
        'options': {
        'defaultType': 'future',
        },
    })

    
    

    coin = current_trade.get_current_coin()
    stake = current_trade.stake
    timeframe = current_trade.timeframe


    print(f"You entered: {coin}")



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




    str_date = (datetime.now()- timedelta(days=days_to_get_candles)).strftime('%b %d,%Y')
    end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')

    df=dataextract(coin,str_date,end_str,timeframe,client)


    x_str = str(df['close'].iloc[-1])
    decimal_index = x_str.find('.')
    round_price = len(x_str) - decimal_index - 1

    current_trade.round_price = round_price

    exchange_info = client.futures_exchange_info()

    for symbol in exchange_info['symbols']:
        if symbol['symbol'] == f"{coin}USDT":
            round_quantity = symbol['quantityPrecision']
            break
    df_copy = df.copy()

    current_trade.round_quantity = round_quantity

    super_df=supertrend_njit(coin, df_copy, period, atr1)
    df_copy = df.copy()
    trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)

    import asyncio
    import json
    import websockets



    df = df[['OpenTime', 'open', 'high', 'low', 'close', 'volume']]

    async def on_message(message,df,current_trade):
        data = json.loads(message)
        coin = current_trade.get_current_coin()
        if data['k']['x'] == True:
            notifier(f'Candle closed : {timeframe} , coin : {coin}')
            df = get_latest_df(data, df)
            df_copy = df.copy()
            super_df=supertrend_njit(coin, df_copy, period, atr1)
            ema = get_ema(super_df,'ema_81')
            
            print(f'Length of super_df : {super_df.shape[0]}')
            df_copy = df.copy()
            signal = get_signal(super_df)
            super_df.to_csv('super_df.csv',index=False,mode='w+')
            current_signal = signal
            prev_signal = get_prev_signal(super_df)

            pivot_st = PivotSuperTrendConfiguration()
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            print(f'Prev PivotSuperTrend signal : {prev_pivot_signal},Prev SuperTrend Signal : {prev_signal}' )

            notifier(f'Previous lowerband : {get_prev_lowerband(super_df)} ,Previous  upperband : {get_prev_upperband(super_df)}')
            notifier(f'Current lowerband : {get_lowerband(super_df)} ,Current  upperband : {get_upperband(super_df)}')
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
                
                #stake = get_stake(super_df,client,risk)
                
                quantity = round(stake/entry, current_trade.round_quantity)
                partial_profit_take = round(quantity/2,current_trade.round_quantity)
                
                change = None
                
                if (current_pivot_signal !=prev_pivot_signal): 
                    change = 'longTerm'
                
                if (current_signal != prev_signal):
                    change = 'shortTerm'
                
                order = Order(coin = coin,
                            entry = entry,
                            quantity = quantity,
                            round_price = current_trade.round_price,
                            change = change,
                            partial_profit_take = partial_profit_take,
                            lowerband = lowerband,
                            upperband = upperband
                            )
                
                notifier(f'round price : {order.round_price}')
                
                if pivot_signal == 'Buy' and signal == 'Buy':  
                    order.make_buy_trade(client)   
                    
                    
                elif pivot_signal == 'Buy' and signal == 'Sell':
                    order.quantity = round(order.quantity/2, current_trade.round_quantity)
                    order.partial_profit_take = round(order.quantity/2,current_trade.round_quantity)
                    order.make_sell_trade(client)
                    
                
                elif pivot_signal == 'Sell' and signal == 'Sell':
                    order.make_sell_trade(client)
                    
                elif pivot_signal == 'Sell' and signal == 'Buy':
                    order.quantity = round(order.quantity/2, current_trade.round_quantity)
                    order.partial_profit_take = round(order.quantity/2,current_trade.round_quantity)
                    order.make_buy_trade(client)
                    
                    
                else:
                    notifier(f'Something is wrong...Debug')
                
    #             notifier(f'Overall trend before passing : {over_all_trend}')
    #             notifier(f'signal : {signal},entry : {entry},stake : {stake},quantity : {quantity}')
                notifier(f'Short Term Trend : {signal} , Pivot SuperTrend signal : {pivot_signal}')

            
        return df



    async def listen(df,current_trade):

        coin = current_trade.get_current_coin()
        # Check if the coin has changed
        notifier(f'Checking coin : {coin}, shared_coin : {shared_coin.value}')
        if coin != shared_coin.value:

            #close current positions
            close_any_open_positions(coin,client)
            cancel_all_open_orders(coin,client)


            coin = shared_coin.value
            current_trade.set_current_coin(coin)
            notifier(f"Coin changed! Now trading {coin}.")
            str_date = (datetime.now()- timedelta(days=days_to_get_candles)).strftime('%b %d,%Y')
            end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')

            df=dataextract(coin,str_date,end_str,timeframe,client)
            x_str = str(df['close'].iloc[-1])
            decimal_index = x_str.find('.')
            round_price = len(x_str) - decimal_index - 1

            current_trade.round_price = round_price

            usdt_leverage,busd_leverage = 25,25

            max_usdt_leverage,max_busd_leverage = get_max_leverage(coin, config.api_key, config.secret_key)

            usdt_leverage = min(usdt_leverage, max_usdt_leverage)
            busd_leverage = min(busd_leverage, max_busd_leverage)

            exchange_info = client.futures_exchange_info()

            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == f"{coin}USDT":
                    round_quantity = symbol['quantityPrecision']
                    break
            df_copy = df.copy()

            current_trade.round_quantity = round_quantity
            
        

            super_df=supertrend_njit(coin, df_copy, period, atr1)
            df_copy = df.copy()
            trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)

            signal = trade_df.iloc[-1]['signal']

            close_any_open_positions(coin,client)
            cancel_all_open_orders(coin,client)
            
            entry =  get_entry(super_df)              
            over_all_trend = get_over_all_trend(coin)
            lowerband = get_lowerband(super_df)
            upperband = get_upperband(super_df)

            ema = get_ema(super_df,'ema_81')
            
            tradeConfig = TradeConfiguration()
            risk = tradeConfig.get_risk(over_all_trend,signal)
            
            #stake = get_stake(super_df,client,risk)
            
            quantity = round(stake/entry, round_quantity)

            partial_profit_take = round(quantity/2,round_quantity)

            order = Order(coin = coin,
                        entry = entry,
                        quantity = quantity,
                        round_price = round_price,
                        change = None,
                        partial_profit_take = partial_profit_take,
                        lowerband = lowerband,
                        upperband = upperband
                        )
            
            notifier(f'round price : {order.round_price}')

            if signal == "Buy":
                order.make_buy_trade(client)  
                notifier(f'Made a buy trade when for {coin}')
            else:
                order.make_sell_trade(client)
                notifier(f'Made a sell trade when for {coin}')

        stream = f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_{timeframe}"
        notifier(f'new stream : {stream}')
        async with websockets.connect(stream) as ws:
            try:
                while True:
                    message = await ws.recv()
                    df = await on_message(message,df,current_trade)
                    if coin != shared_coin.value:
                        break
                    
            except websockets.ConnectionClosed:
                print("WebSocket connection closed. Attempting to reconnect...")
                await asyncio.sleep(10)
                df = await listen(df)
        return df
    
    


  

    while True:
        try:
            notifier(f'Old coin : {coin}')
            df = await listen(df,current_trade)
        except Exception as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            await asyncio.sleep(10)



# Run the asyncio event loop

def run_async_main(shared_coin,current_trade):
        asyncio.run(main(shared_coin,current_trade))



if __name__ == "__main__":
    
    coin = input("Please enter the coin name: ")
    stake = float(input("Enter the stake :"))

    timeframe = get_timeframe()
    print(f"You've selected {timeframe}. Please reconfirm.")

    reconfirm = get_timeframe()

    if timeframe == reconfirm:
        print(f"Thank you! Your timeframe of {timeframe} has been confirmed.")
    else:
        print("The selections don't match. Please try again.")

    

    current_trade = CurrentTrade(coin=coin,timeframe=timeframe,stake=stake)
    manager = Manager()
    shared_coin = manager.Value(str, coin)
    shared_coin.value = coin

    notifier_with_photo("data/saravanabhava.jpeg", "SARAVANA BHAVA")

    p1 = Process(target=fetch_volatile_coin, args=(shared_coin,))
    p2 = Process(target=run_async_main, args=(shared_coin,current_trade))


    p1.start()
    p2.start()
    p1.join()
    p2.join()

