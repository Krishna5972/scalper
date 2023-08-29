import pandas as pd
import websocket
import json
import time
from threading import Timer
import pickle

stream = "wss://fstream.binance.com/ws/!ticker@arr"

symbol_data = {}

def get_volatile_dataframe():
    # Convert the dictionary to DataFrame
    df = pd.DataFrame.from_dict(symbol_data, orient='index')

    df['PctRange'] = (df['h'] - df['l']) / abs(df['l']) * 100
    df['Volatility'] = df['PctRange'].rolling(window=1).mean()
    df_volatility = df.sort_values(by='Volatility')
    
    return df_volatility


def on_message(ws, message):
    msg = json.loads(message)
    symbols = [x for x in msg if x['s'].endswith('USDT')]
    frame = pd.DataFrame(symbols)[['E', 's', 'o', 'h', 'l', 'c']]
    frame.E = pd.to_datetime(frame.E, unit='ms')
    frame[['o', 'h', 'l', 'c']] = frame[['o', 'h', 'l', 'c']].astype(float)

    symbol_dataframes = {}  # Dictionary to store each symbol's dataframe

    for _, row in frame.iterrows():
        symbol = row['s']
        symbol_data[symbol] = row.to_dict()

    

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed.")

def stop_ws(ws):
    print("Stopping websocket after 1 minute.")
    ws.close()

ws = websocket.WebSocketApp(stream, on_message=on_message, on_error=on_error, on_close=on_close)

while True:
    try:
        timer = Timer(6, stop_ws, [ws])  # Set a timer to close websocket after 1 minute
        timer.start()
        
        
        ws.run_forever()

        df = get_volatile_dataframe()
        volatile_coin = df.iloc[-1]['s']

        print(volatile_coin)

        with open("volatile_coin.pkl", "wb") as file:
            pickle.dump(volatile_coin, file)

        print(f'Sleeping for one hour')

        time.sleep(10)
        
        print('Listeneing again...')
        # After websocket closes, you can include a delay here if desired 
        # (e.g., `time.sleep(10)` for a 10-second delay before the next connection).
    except Exception as e:
        print(f"Error: {e}. Retrying in 10 seconds...")
        time.sleep(10)
