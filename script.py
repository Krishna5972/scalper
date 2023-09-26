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
    use_sl = current_trade.use_sl

    print(f"You entered: {coin}")



    usdt_leverage,busd_leverage = 1,1

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
        use_sl = current_trade.use_sl
        #now = datetime.utcnow()
        # if now.hour == 23 and now.minute == 59:
        #     close_any_open_positions(coin,client)
        #     cancel_all_open_orders(coin,client)

        if data['k']['x'] == True:
            notifier(f'Candle closed : {timeframe} , coin : {coin}')
            df = get_latest_df(data, df)
            df_copy = df.copy()

            pivot_st = PivotSuperTrendConfiguration()
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            super_df = pivot_super_df

            #super_df=supertrend_njit(coin, df_copy, period, atr1)
            ema = get_ema(super_df,'ema_81')
            
            print(f'Length of super_df : {super_df.shape[0]}')
            df_copy = df.copy()
            signal = get_signal(super_df)
            #super_df.to_csv('super_df.csv',index=False,mode='w+')
            current_signal_short = signal
            prev_signal_short = get_prev_signal(super_df)

            trade_df = create_signal_df(super_df,df,coin,timeframe,atr1,period,100,100)

            
            pivot_st = PivotSuperTrendConfiguration(period = 2, atr_multiplier = 2.6, pivot_period = 2)
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            super_df = pivot_super_df

            signal_long = get_signal(super_df)
            current_signal_long = signal_long
            prev_signal_long = get_prev_signal(super_df)

            
            if (current_signal_short != prev_signal_short) or (current_signal_long != prev_signal_long): 
                
                close_any_open_positions(coin,client)
                cancel_all_open_orders(coin,client)
                middle_dc = get_middle_dc(client,coin)
                entry =  get_entry(super_df)              
                #over_all_trend = get_over_all_trend(coin)
                lowerband = get_lowerband(super_df)
                upperband = get_upperband(super_df)


       
                tradeConfig = TradeConfiguration()
                risk = 0.01
                
                #stake = get_stake(super_df,client,risk)
                
                quantity = round(stake/entry, current_trade.round_quantity)
                partial_profit_take = round(quantity/2,current_trade.round_quantity) 
                change = None
                
                if (current_pivot_signal !=prev_pivot_signal): 
                    change = 'longTerm'
                
                if (current_signal_short != prev_signal_short):
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
                       
                if current_signal_short == 'Buy' and current_signal_long == 'Buy':       
                    order.make_buy_trade(client) 
                    notifier(f'ShortTerm : Buy , LongTerm : Buy => Bought')  

                elif current_signal_short == 'Sell' and current_signal_long == 'Buy':
                    order.make_buy_trade(client) 
                    notifier(f'ShortTerm : Sell , LongTerm : Buy => Bought')
                    
                # elif pivot_signal == 'Buy' and signal == 'Sell':
                #     order.quantity = round(order.quantity/2, current_trade.round_quantity)
                #     order.partial_profit_take = round(order.quantity/2,current_trade.round_quantity)
                #     order.make_sell_trade(client)
                    
                
                elif current_signal_short == 'Sell' and current_signal_long == 'Sell':   
                    order.make_sell_trade(client)
                    notifier(f'ShortTerm : Sell , LongTerm : Sell => Sold')

                elif current_signal_short == 'Buy' and current_signal_long == 'Sell':
                    order.make_sell_trade(client)
                    notifier(f'ShortTerm : Buy , LongTerm : Sell => Sold')

                    
                # elif pivot_signal == 'Sell' and signal == 'Buy':
                #     order.quantity = round(order.quantity/2, current_trade.round_quantity)
                #     order.partial_profit_take = round(order.quantity/2,current_trade.round_quantity)
                #     order.make_buy_trade(client)     
                else:
                    notifier(f'Something is wrong...Debug')      
            
            
               


        
        
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
                
                pivot_st = PivotSuperTrendConfiguration()

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

        TIMEOUT_SECONDS = 30

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
    stake = float(input("Enter the stake :"))
    check_for_volatilte_coin = int(input("Please enter 1 to trade most volatile coin always: "))

    timeframe = get_timeframe()
    print(f"You've selected {timeframe}. Please reconfirm.")

    reconfirm = get_timeframe()

    if timeframe == reconfirm:
        print(f"Thank you! Your timeframe of {timeframe} has been confirmed.")
    else:
        print("The selections don't match. Please try again.")

    

    current_trade = CurrentTrade(coin=coin,timeframe=timeframe,stake=stake,check_for_volatilte_coin=check_for_volatilte_coin,use_sl = 0)
    manager = Manager()
    shared_coin = manager.Value(str, coin)
    shared_coin.value = coin

    notifier_with_photo("data/saravanabhava.jpeg", "SARAVANA BHAVA")

    

    #p1 = Process(target=get_coin, args=(shared_coin,))
    p2 = Process(target=run_async_main, args=(shared_coin,current_trade))


    #p1.start()
    p2.start()
   # p1.join()
    p2.join()

if __name__ == "__main__":
    main_execution()
    
    

