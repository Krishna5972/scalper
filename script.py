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



    usdt_leverage,busd_leverage = 1,1

    max_usdt_leverage,max_busd_leverage = get_max_leverage(coin, config.api_key, config.secret_key)

    usdt_leverage = max(usdt_leverage, max_usdt_leverage)
    busd_leverage = max(busd_leverage, max_busd_leverage)

    is_usdt_exist=1
    is_busd_exist=1

    

    change_leverage(coin,max_usdt_leverage,max_busd_leverage)

    client._create_futures_api_uri = create_futures_api_uri_v1.__get__(client, Client)




    str_date = (datetime.now()- timedelta(days=days_to_get_candles)).strftime('%b %d,%Y')
    end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')

    df=dataextract(coin,str_date,end_str,timeframe,client)

    df= df.tail(330).reset_index(drop=True)


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

    try:
        current_trade.round_quantity = round_quantity
    except UnboundLocalError as e:
        current_trade.round_quantity = 0
        notifier('Could not find quantityPrecision')


    pivot_st = PivotSuperTrendConfiguration()
    super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
    
    df_copy = df.copy()
    trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)

    import asyncio
    import json
    import websockets



    df = df[['OpenTime', 'open', 'high', 'low', 'close', 'volume']]

    async def on_message(message,df,current_trade):
        data = json.loads(message)
        coin = current_trade.get_current_coin()
        #now = datetime.utcnow()
        # if now.hour == 23 and now.minute == 59:
        #     close_any_open_positions(coin,client)
        #     cancel_all_open_orders(coin,client)

        if data['k']['x'] == True:
            
            df = get_latest_df(data, df)
            if df.shape[0] < 40:
                return df
            df_copy = df.copy()

            pivot_st = PivotSuperTrendConfiguration(period = 1, atr_multiplier = 1, pivot_period = 1)
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            super_df = pivot_super_df

            upperband_1 = pivot_super_df.iloc[-1]['upperband']
            lowerband_1 = pivot_super_df.iloc[-1]['lowerband']

            #super_df=supertrend_njit(coin, df_copy, period, atr1)
            #ema = get_ema(super_df,'ema_81')
            
            df_copy = df.copy()
            signal = get_signal(super_df)
            #super_df.to_csv('super_df.csv',index=False,mode='w+')
            current_signal_short = signal
            prev_signal_short = get_prev_signal(super_df)

            #trade_df = create_signal_df(super_df,df,coin,timeframe,atr1,period,100,100)

            
            pivot_st = PivotSuperTrendConfiguration(period = 2, atr_multiplier = 2.6, pivot_period = 2)
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            super_df = pivot_super_df

            signal_long = get_signal(super_df)
            current_signal_long = signal_long
            prev_signal_long = get_prev_signal(super_df)

            str_date = (datetime.now()- timedelta(days=3)).strftime('%b %d,%Y')
            df_15m=dataextract(coin,str_date,end_str,'15m',client)

            df_15m = df_15m.tail(330).reset_index(drop=True)


            df_15m = df_15m.iloc[:-1]

            pivot_super_df_15m = supertrend_pivot(coin, df_15m, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            long_signal_15m = get_pivot_supertrend_signal(pivot_super_df_15m)

            long_signal_15m_prev = get_prev_signal(pivot_super_df_15m)

            

            upperband_2_6 = pivot_super_df.iloc[-1]['upperband']
            lowerband_2_6= pivot_super_df.iloc[-1]['lowerband']

            upperband_15m = pivot_super_df_15m.iloc[-1]['upperband']
            lowerband_15m = pivot_super_df_15m.iloc[-1]['lowerband']


            notifier(f"""Candle closed : {timeframe} , coin : {coin}
                      1 [upperband : {upperband_1} , lowerband : {lowerband_1}] 
                      2.6 [upperband : {upperband_2_6} , lowerband : {lowerband_2_6}] 
                      15m : [upperband : {upperband_15m} , lowerband : {lowerband_15m}]""")
            
            if (current_signal_short != prev_signal_short) or (current_signal_long != prev_signal_long) or ((long_signal_15m != long_signal_15m_prev ) and (datetime.now().minute in [0,15,30,45])): 
                
                close_any_open_positions(coin,client)
                cancel_all_open_orders(coin,client)
                #middle_dc = get_middle_dc(client,coin)
                entry =  get_entry(super_df)              
                #over_all_trend = get_over_all_trend(coin)
                lowerband = get_lowerband(super_df)
                upperband = get_upperband(super_df)
                
                #stake = get_stake(super_df,client,risk)
                
                quantity = round(stake/entry, current_trade.round_quantity)
                partial_profit_take = round(quantity/2,current_trade.round_quantity) 
                


          
                
                order = Order(coin = coin,
                            entry = entry,
                            quantity = quantity,
                            round_price = current_trade.round_price,
                            change = None,
                            partial_profit_take = partial_profit_take,
                            lowerband = lowerband,
                            upperband = upperband
                            )
                       
                if long_signal_15m != long_signal_15m_prev:
                    if long_signal_15m == "Buy":
                        order.make_buy_trade(client)
                        notifier(f'Long15m : Buy => Bought')
                    elif long_signal_15m == "Sell":
                        order.make_sell_trade(client)
                        notifier(f'Long15m : Sell => Sold')

                elif current_signal_short == 'Sell' and current_signal_long == 'Buy' and long_signal_15m =='Buy':
                    order.make_buy_trade(client) 
                    notifier(f'ShortTerm : Sell , LongTerm : Buy , Long15m : Buy => Bought')
              

                elif current_signal_short == 'Buy' and current_signal_long == 'Sell' and long_signal_15m =='Sell':
                    order.quantity =  round(order.quantity/2, round_quantity)
                    order.partial_profit_take = round(order.partial_profit_take/2, round_quantity) 
                    order.make_sell_trade(client)
                    notifier(f'ShortTerm : Buy , LongTerm : Sell  , Long15m : Sell => Sold')
                


                    
                # elif pivot_signal == 'Sell' and signal == 'Buy':
                #     order.quantity = round(order.quantity/2, current_trade.round_quantity)
                #     order.partial_profit_take = round(order.quantity/2,current_trade.round_quantity)
                #     order.make_buy_trade(client)     
                else:
                    notifier(f'Waiting Patiently to strike.....')      
            
            
               


        
        
        return df


    async def on_error(ws, error):
        notifier(f"WebSocket Error: {error} {coin}")

    async def on_close(ws, close_status_code, close_msg):
        notifier(f"WebSocket connection closed. {coin}" )

    async def stop_ws(ws):
        print(f"Stopping websocket after {60} seconds. {coin}")
        ws.close()

    async def listen(df,current_trade):
        check_for_volatilte_coin = current_trade.check_for_volatilte_coin

        coin = current_trade.get_current_coin()
        # Check if the coin has changed
        notifier(f'Checking coin : {coin}, shared_coin : {shared_coin.value}')
        if check_for_volatilte_coin == 1:
            use_sl = 0
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

                df = df.tail(330).reset_index(drop=True)
                x_str = str(df['close'].iloc[-1])
                decimal_index = x_str.find('.')
                round_price = len(x_str) - decimal_index - 1

                current_trade.round_price = round_price

                usdt_leverage,busd_leverage = 25,25

                max_usdt_leverage,max_busd_leverage = get_max_leverage(coin, config.api_key, config.secret_key)

                usdt_leverage = max(usdt_leverage, max_usdt_leverage)
                busd_leverage = max(busd_leverage, max_busd_leverage)

                exchange_info = client.futures_exchange_info()

                for symbol in exchange_info['symbols']:
                    if symbol['symbol'] == f"{coin}USDT":
                        round_quantity = symbol['quantityPrecision']
                        break
                df_copy = df.copy()

                current_trade.round_quantity = round_quantity
                
                pivot_st = PivotSuperTrendConfiguration(period = 1, atr_multiplier = 1, pivot_period = 1)

                super_df=supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
                df_copy = df.copy()
                trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)

                signal = trade_df.iloc[-1]['signal']

                close_any_open_positions(coin,client)
                cancel_all_open_orders(coin,client)
                
                entry =  get_entry(super_df)              
                over_all_trend = get_over_all_trend(coin)
                lowerband = get_lowerband(super_df)
                upperband = get_upperband(super_df)

                
                #stake = get_stake(super_df,client,risk)
                
                quantity = round((stake/2)/entry, round_quantity)

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
                    order.make_buy_trade(client,big_profit = 1)  
                    notifier(f'Made a buy trade for {coin} looking for BIG Profit')
                else:
                    order.quantity =  round(order.quantity/2, round_quantity)
                    order.partial_profit_take = round(order.partial_profit_take/2, round_quantity) 
                    order.make_sell_trade(client,big_profit =1)
                    notifier(f'Made a sell trade for {coin} for BIG Profit')

        TIMEOUT_SECONDS = 60

        stream = f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_{timeframe}"
        notifier(f'new stream : {stream}')
        async with websockets.connect(stream) as ws:
            try:
                while True:
                    message = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECONDS)
                    df = await on_message(message,df,current_trade)
                    if check_for_volatilte_coin == 1:
                        if coin != shared_coin.value:
                            break
                    
                    
            except asyncio.TimeoutError:
                notifier(f"No message received for the past 30 seconds! for {coin}") 
                await asyncio.sleep(10)
                df = await listen(df,current_trade)

            except websockets.ConnectionClosed:
                notifier("WebSocket connection closed. Attempting to reconnect...")
                await asyncio.sleep(10)
                df = await listen(df,current_trade)
        return df
    
    


  

    while True:
        try:
            #notifier(f'Old coin : {coin}')
            str_date = (datetime.now()- timedelta(days=days_to_get_candles)).strftime('%b %d,%Y')
            end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')
            df=dataextract(coin,str_date,end_str,timeframe,client)
            df= df.tail(330).reset_index(drop=True)

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

            try:
                current_trade.round_quantity = round_quantity
            except UnboundLocalError as e:
                current_trade.round_quantity = 0
                notifier('Could not find quantityPrecision')


            df = await listen(df,current_trade)
        except Exception as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            notifier(f"Error: {e}. Retrying in 10 seconds...")
            await asyncio.sleep(10)



# Run the asyncio event loop

def run_async_main(shared_coin,current_trade):
        asyncio.run(main(shared_coin,current_trade))



def main_execution():
    coin = input("Please enter the coin name: ")
    coin = coin.upper()
    stake = 600
    check_for_volatilte_coin = 1

    timeframe = '5m'
    print(f"Your timeframe of {timeframe} has been confirmed.")

    current_trade = CurrentTrade(coin=coin,timeframe=timeframe,stake=stake,check_for_volatilte_coin=check_for_volatilte_coin,use_sl = 0)
    manager = Manager()
    shared_coin = manager.Value(str, coin)
    shared_coin.value = coin

    notifier_with_photo("data/saravanabhava.jpeg", "SARAVANA BHAVA")

    

    p1 = Process(target=get_most_volatile_coin_d, args=(shared_coin,))
    p2 = Process(target=run_async_main, args=(shared_coin,current_trade))


    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == "__main__":
    main_execution()
    
    

