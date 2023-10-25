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


async def main(shared_coin,current_trade,master_order_history):

     
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
            notifier(f'Precesion for coin : {coin} : {round_quantity}')
            break
    

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

    async def on_message(message,df,current_trade,master_order_history):
        data = json.loads(message)
        coin = current_trade.get_current_coin()
        if data['k']['x'] == True:
            
            df = get_latest_df(data, df)
            if df.shape[0] < 40:
                notifier(f'{coin} has less than 40 candles not trading')
                return df
            df_copy = df.copy()

            pivot_st = PivotSuperTrendConfiguration(period = 1, atr_multiplier = 1, pivot_period = 1)
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            super_df = pivot_super_df

            notifier(f'short term Previous lowerband : {get_prev_lowerband(super_df)} ,Previous  upperband : {get_prev_upperband(super_df)}')
            notifier(f'short term Current lowerband : {get_lowerband(super_df)} ,Current  upperband : {get_upperband(super_df)}')

            print(f'Len of shorttem {pivot_super_df.shape[0]}')

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

            
            pivot_st = PivotSuperTrendConfiguration(period = 2, atr_multiplier = 2.8, pivot_period = 2)
            pivot_super_df = supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
            pivot_signal = get_pivot_supertrend_signal(pivot_super_df)
            current_pivot_signal = pivot_signal
            prev_pivot_signal = get_prev_pivot_supertrend_signal(pivot_super_df)

            super_df = pivot_super_df

            signal_long = get_signal(super_df)
            current_signal_long = signal_long
            prev_signal_long = get_prev_signal(super_df)

            
           


            trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)

            prev_trade_perc = trade_df.iloc[-2]['percentage']

            limit_stake = 0

            if prev_trade_perc > 0:
                limit_stake = 1


            notifier(f'Candle closed : {timeframe}')

            notifier(f'Previous lowerband : {get_prev_lowerband(super_df)} ,Previous  upperband : {get_prev_upperband(super_df)}')
            notifier(f'Current lowerband : {get_lowerband(super_df)} ,Current  upperband : {get_upperband(super_df)}')

            
            if (current_signal_short != prev_signal_short) or (current_signal_long != prev_signal_long): 

                if current_signal_long != prev_signal_long:
                    close_any_open_positions(coin,client)
                    cancel_all_open_orders(coin,client)
                    notifier('Long term trend changed closing positions and all open trades')
                    master_order_history = {}

                #if tp limit order is not reached, the dca orders will still be here, so cancel them.
                dca_order_ids = []
      

                for coin, order_types in master_order_history.items():
                    for order_type, take_profits in order_types.items():
                        for take_profit in list(take_profits.keys()):
                            for limit_order_id in list(take_profits[take_profit].keys()):
                                dca_order_id = take_profits[take_profit][limit_order_id]
                                dca_order_ids.append([dca_order_id,take_profit])
                                del take_profits[take_profit][limit_order_id]
                                if not take_profits[take_profit]:
                                    del take_profits[take_profit]



                #before cancelling checking if dca is reached to increase tp position

               
                account_history = client.futures_account_trades(limit=100)
                account_orders = pd.DataFrame(account_history)

                account_order_history_dict = {}
                for index, row in account_orders.iterrows():
                    order_id = row['orderId']
                    side = row['side']
                    qty = row['qty']
                    account_order_history_dict[order_id] = {'side': side, 'qty': qty}


                # {
                #     'id' : {
                #         'side' : "BUY",
                #         'qty' : 0.16
                #     }
                # }


                
                if len(dca_order_ids) > 0:
                    for idx,order_id in enumerate(dca_order_ids):  #it means dca has hit for trend so i need to increase the profit limit
                        if order_id[0] in  list(account_orders['orderId']):
                            side = account_order_history_dict[order_id[0]]['side']
                            qty = account_order_history_dict[order_id[0]]['qty']
                            price = order_id[1]

                            if side == 'BUY':
                                client.futures_create_order(
                                    symbol=f'{coin}USDT',
                                    price=round(price,current_trade.round_price),
                                    side='SELL',
                                    positionSide='LONG',
                                    quantity=float(qty),
                                    timeInForce='GTC',
                                    type='LIMIT',
                                    # reduceOnly=True,cc
                                    closePosition=False,
                                    # stopPrice=round(take_profit,2),
                                    workingType='MARK_PRICE',
                                    priceProtect=True
                                )
                            else:
                                client.futures_create_order(
                                    symbol=f'{coin}USDT',
                                    price=round(price,current_trade.round_price),
                                    side='BUY',
                                    positionSide='SHORT',
                                    quantity=float(qty),
                                    timeInForce='GTC',
                                    type='LIMIT',
                                    # reduceOnly=True,
                                    closePosition=False,
                                    # stopPrice=round(take_profit,2),
                                    workingType='MARK_PRICE',
                                    priceProtect=True
                               )







                if len(dca_order_ids) > 0:
                    for order_id in dca_order_ids:
                        try:
                            client.futures_cancel_order(symbol=f'{coin}USDT', orderId=order_id[0])
                            notifier(f'Order id : DCA order : {order_id[0]} is cancelled as trend changed')
                        except Exception as e:
                            notifier('DCAed so cannot cannel as order is already filled')

            

                entry =  get_entry(super_df)              

                
                quantity = round(stake/entry, current_trade.round_quantity)
                partial_profit_take = round(quantity/2,current_trade.round_quantity) 
                
                if current_signal_short == "Sell": # Do the inverse
                    take_profit = upperband_1
                else:
                    take_profit = lowerband_1

          
                
                order = Order(coin = coin,
                            entry = entry,
                            quantity = quantity,
                            round_price = current_trade.round_price,
                            change = None,
                            take_profit  = take_profit,
                            lowerband = lowerband_1,
                            upperband = upperband_1,
                            master_order_history = master_order_history
                            )
                       
                notifier(f'Entry for coin {coin} {entry}')

                if current_signal_short == 'Sell' and current_signal_long == 'Buy' and limit_stake == 0:
                    order.make_buy_trade(client) 
                    notifier(f'ShortTerm : Sell , LongTerm : Buy , Long15m : Buy => Bought')

                elif current_signal_short == 'Sell' and current_signal_long == 'Buy' and limit_stake ==1:
                    order.quantity = round(order.quantity/2,current_trade.round_quantity)
                    order.make_buy_trade(client) 
                    notifier(f'ShortTerm : Sell , LongTerm : Buy , Long15m : Sell => Bought with less amount')

                elif current_signal_short == 'Buy' and current_signal_long == 'Sell' and limit_stake == 0:
                    order.make_sell_trade(client)
                    notifier(f'ShortTerm : Buy , LongTerm : Sell  , Long15m : Sell => Sold')

                elif current_signal_short == 'Buy' and current_signal_long == 'Sell' and limit_stake == 1:
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
                x_str = str(df['close'].iloc[-1])
                decimal_index = x_str.find('.')
                round_price = len(x_str) - decimal_index - 1

                current_trade.round_price = round_price

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
                
                pivot_st = PivotSuperTrendConfiguration(period = 1, atr_multiplier = 1, pivot_period = 1)

                super_df=supertrend_pivot(coin, df_copy, pivot_st.period, pivot_st.atr_multiplier, pivot_st.pivot_period)
                df_copy = df.copy()
                trade_df=create_signal_df(super_df,df_copy,coin,timeframe,atr1,period,100,100)


                close_any_open_positions(coin,client)
                cancel_all_open_orders(coin,client)
                
        
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
                    df = await on_message(message,df,current_trade,master_order_history)
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
                    
                    high = df.iloc[-1]['high']
                    low = df.iloc[-1]['low']

                    dca_orders = []
   
        
                    if current_trade.coin in master_order_history and 'Buy' in master_order_history[current_trade.coin]:
                        buy_dict = master_order_history[current_trade.coin]['Buy']
                        for rounded_price in list(buy_dict.keys()):  # iterate over a copy of the keys
                            if high > rounded_price:
                                dca_orders.extend(buy_dict[rounded_price].values())
                                del buy_dict[rounded_price]  # remove the key-value pair from the dictionary
                               
                    if current_trade.coin in master_order_history and 'Sell' in master_order_history[current_trade.coin]:
                        buy_dict = master_order_history[current_trade.coin]['Sell']
                        for rounded_price in list(buy_dict.keys()):  # iterate over a copy of the keys
                            if low < rounded_price:
                                dca_orders.extend(buy_dict[rounded_price].values())
                                del buy_dict[rounded_price]  # remove the key-value pair from the dictionary
                               
                    if len(dca_orders) > 0:
                        for order_id in dca_orders:
                            try:
                                client.futures_cancel_order(symbol=f'{coin}USDT', orderId=order_id)
                                notifier(f'Order id : {order_id} is cancelled')
                            except Exception as e:
                                notifier(f'DCAed so cannot cancel the order')
                            
                            


        


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
            
            if current_trade.stream == 'futures':
                df=dataextract(coin,str_date,end_str,timeframe,client,futures=1)
            else:
                df=dataextract(coin,str_date,end_str,timeframe,client,futures=0)
        
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

def run_async_main(shared_coin,current_trade,master_order_history):
        asyncio.run(main(shared_coin,current_trade,master_order_history))



def main_execution():
    coin = input("Please enter the coin name: ")
    coin = coin.upper()
    stake = 30
    check_for_volatilte_coin = 1
    master_order_history = {}

    timeframe = '5m'
    print(f"Your timeframe of {timeframe} has been confirmed.")

    current_trade = CurrentTrade(coin=coin,timeframe=timeframe,stake=stake,check_for_volatilte_coin=check_for_volatilte_coin,use_sl = 0)
    manager = Manager()
    shared_coin = manager.Value(str, coin)
    shared_coin.value = coin

    notifier_with_photo("data/saravanabhava.jpeg", "SARAVANA BHAVA")

    

    p1 = Process(target=get_most_volatile_coin_d, args=(shared_coin,))
    p2 = Process(target=run_async_main, args=(shared_coin,current_trade,master_order_history))


    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == "__main__":
    main_execution()
    
    

