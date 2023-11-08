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
import  trade_config


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


    change_leverage(coin,max_usdt_leverage,max_busd_leverage)




    str_date = (datetime.now()- timedelta(days=days_to_get_candles)).strftime('%b %d,%Y')
    end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')

    df=dataextract(coin,str_date,end_str,timeframe,client,futures=1)

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

    stream = get_stream(coin, timeframe)
    current_trade.is_spot = is_spot_available(coin)
    if 'fstream' in stream:
        current_trade.stream = 'futures'
    else:
        current_trade.stream = 'spot'

    
    
    if current_trade.stream == 'futures':
        df=dataextract(coin,str_date,end_str,timeframe,client,futures=1)
    else:
        df=dataextract(coin,str_date,end_str,timeframe,client,futures=0)

    
    df = df.tail(330).reset_index(drop=True)
    

    try:
        current_trade.round_quantity = round_quantity
    except UnboundLocalError as e:
        current_trade.round_quantity = 0
        notifier('Could not find quantityPrecision')

    if get_funding(coin) > -0.005:
        df=dataextract(coin,str_date,end_str,timeframe,client)

        df= df.tail(330).reset_index(drop=True)
    df_copy = df.copy()
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
        if data['k']['x'] == True:
            
            df = get_latest_df(data, df)
            if df.shape[0] < 40:
                notifier(f'{coin} has less than 40 candles not trading')
                return df
            df_copy = df.copy()

            pivot_st = PivotSuperTrendConfiguration(period = trade_config.short_term_period, 
                                                    atr_multiplier = trade_config.short_term_atr_multiplier, 
                                                    pivot_period = trade_config.short_term_pivot_period)
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            super_df = pivot_super_df

            # notifier(f'short term Previous lowerband : {get_prev_lowerband(super_df)} ,Previous  upperband : {get_prev_upperband(super_df)}')
            # notifier(f'short term Current lowerband : {get_lowerband(super_df)} ,Current  upperband : {get_upperband(super_df)}')

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

            
            pivot_st = PivotSuperTrendConfiguration(period = trade_config.long_term_period,
                                                     atr_multiplier = trade_config.long_term_atr_multiplier, 
                                                     pivot_period = trade_config.long_term_pivot_period)     
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


            pivot_super_df_15m = supertrend_pivot(coin, df_15m, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            long_signal_15m = get_pivot_supertrend_signal(pivot_super_df_15m)

            long_signal_15m_prev = get_prev_signal(pivot_super_df_15m)

            

            upperband_2_6 = pivot_super_df.iloc[-1]['upperband']
            lowerband_2_6= pivot_super_df.iloc[-1]['lowerband']

            upperband_15m = pivot_super_df_15m.iloc[-1]['upperband']
            lowerband_15m = pivot_super_df_15m.iloc[-1]['lowerband']

            notifier(f'Candle closed {coin}: {timeframe} Stream : {current_trade.stream}')

            # notifier(f'Previous lowerband : {get_prev_lowerband(super_df)} ,Previous  upperband : {get_prev_upperband(super_df)}')
            # notifier(f'Current lowerband : {get_lowerband(super_df)} ,Current  upperband : {get_upperband(super_df)}')

            
            if (current_signal_short != prev_signal_short) or (current_signal_long != prev_signal_long): 
                
                close_any_open_positions(coin,client)
                cancel_all_open_orders(coin,client)
                #middle_dc = get_middle_dc(client,coin)
                entry =  get_entry(super_df)              
                #over_all_trend = get_over_all_trend(coin)
                lowerband = get_lowerband(super_df)
                upperband = get_upperband(super_df)
                
                time_now = datetime.utcnow()
                hour = time_now.hour

                if hour > 12:
                    notifier('Reducing the stake as its after 12 UTC')
                    stake = trade_config.stake/2
                else:  
                    stake = trade_config.stake

                
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
                       

                if current_signal_short == 'Sell' and current_signal_long == 'Buy' and long_signal_15m =='Buy':
                    order.make_buy_trade(client) 
                    notifier(f'ShortTerm : Sell , LongTerm : Buy , Long15m : Buy => Bought')

                elif current_signal_short == 'Sell' and current_signal_long == 'Buy' and long_signal_15m =='Sell':
                    order.quantity = round(order.quantity/2,current_trade.round_quantity)
                    order.make_buy_trade(client) 
                    notifier(f'ShortTerm : Sell , LongTerm : Buy , Long15m : Sell => Bought with less amount')

                elif current_signal_short == 'Buy' and current_signal_long == 'Sell' and long_signal_15m =='Sell':
                    order.make_sell_trade(client)
                    notifier(f'ShortTerm : Buy , LongTerm : Sell  , Long15m : Sell => Sold')

                elif current_signal_short == 'Buy' and current_signal_long == 'Sell' and long_signal_15m =='Buy':
                    order.quantity = round(order.quantity/2,current_trade.round_quantity)
                    order.make_sell_trade(client)
                    notifier(f'ShortTerm : Buy , LongTerm : Sell  , Long15m : Buy => Sold with less amount')
                


                    
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
                current_trade.coin = coin
                notifier(f"Coin changed! Now trading {coin}.")
                str_date = (datetime.now()- timedelta(days=days_to_get_candles)).strftime('%b %d,%Y')
                end_str = (datetime.now() +  timedelta(days=3)).strftime('%b %d,%Y')

                df=dataextract(coin,str_date,end_str,timeframe,client,futures=1)

                
                x_str = str(df['close'].iloc[-1])
                decimal_index = x_str.find('.')
                round_price = len(x_str) - decimal_index - 1

                current_trade.round_price = round_price

                stream = get_stream(coin, timeframe)
                current_trade.is_spot = is_spot_available(coin)
                if 'fstream' in stream:
                    current_trade.stream = 'futures'
                else:
                    current_trade.stream = 'spot'

                
                
                if current_trade.stream == 'futures':
                    df=dataextract(coin,str_date,end_str,timeframe,client,futures=1)
                else:
                    df=dataextract(coin,str_date,end_str,timeframe,client,futures=0)

                
                df = df.tail(330).reset_index(drop=True)

                usdt_leverage,busd_leverage = 25,25

                max_usdt_leverage,max_busd_leverage = get_max_leverage(coin, config.api_key, config.secret_key)

                usdt_leverage = max(usdt_leverage, max_usdt_leverage)
                busd_leverage = max(busd_leverage, max_busd_leverage)

                change_leverage(coin,max_usdt_leverage,max_busd_leverage)

                exchange_info = client.futures_exchange_info()

                for symbol in exchange_info['symbols']:
                    if symbol['symbol'] == f"{coin}USDT":
                        round_quantity = symbol['quantityPrecision']
                        break
                if get_funding(coin) > -0.0005:
                    df=dataextract(coin,str_date,end_str,timeframe,client)

                    df = df.tail(330).reset_index(drop=True)
                df_copy = df.copy()

                current_trade.round_quantity = round_quantity
                
                pivot_st = PivotSuperTrendConfiguration(period = trade_config.short_term_period,
                                                         atr_multiplier = trade_config.short_term_atr_multiplier, 
                                                         pivot_period = trade_config.short_term_pivot_period)

                super_df=supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
                df_copy = df.copy()
                trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)

                
        
        stream = get_stream(coin, timeframe)
        current_trade.is_spot = is_spot_available(coin)
        if 'fstream' in stream:
            current_trade.stream = 'futures'
        else:
            current_trade.stream = 'spot'

        TIMEOUT_SECONDS = 60

        notifier(f'new stream : {stream}')
        funding_check = 0
        async with websockets.connect(stream) as ws:
            try:
                while True:
                    message = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECONDS)
                    df = await on_message(message,df,current_trade)
                    if check_for_volatilte_coin == 1:
                        if coin != shared_coin.value:
                            break
                    if funding_check > 600:
                        funding = get_funding(coin)
                        if current_trade.stream == 'spot' and funding < -0.005:
                            notifier(f'{coin} funding rate increased so connecting to futures stream')
                            break
                        elif current_trade.stream == 'futures' and funding > -0.005 and current_trade.is_spot == 1:
                            notifier(f'{coin} funding rate decreased so connecting to spot stream if it exists')

                            break
                        
                        funding_check = 0
                    
                    funding_check += 1

                    
                    
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

            coin = current_trade.get_current_coin()

            notifier(f'Met with an error now getting data for coin : {coin}')

            stream = get_stream(coin, timeframe)
            current_trade.is_spot = is_spot_available(coin)
            if 'fstream' in stream:
                current_trade.stream = 'futures'
            else:
                current_trade.stream = 'spot'


            df=dataextract(coin,str_date,end_str,timeframe,client,futures=1)

            x_str = str(df['close'].iloc[-1])
            decimal_index = x_str.find('.')
            round_price = len(x_str) - decimal_index - 1

            current_trade.round_price = round_price
            
            
            if current_trade.stream == 'futures':
                df=dataextract(coin,str_date,end_str,timeframe,client,futures=1)
            else:
                df=dataextract(coin,str_date,end_str,timeframe,client,futures=0)
        
            df= df.tail(330).reset_index(drop=True)

           

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
    with open("volatile_coin.pkl", "rb") as file:
        loaded_volatile_coin = pickle.load(file)
    coin = loaded_volatile_coin
    coin = coin.upper()
    stake = trade_config.stake
    check_for_volatilte_coin = 1

    timeframe = trade_config.timeframe
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
    
    

