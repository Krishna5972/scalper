import json
import websocket
from threading import Timer
import sys
sys.path.append('..')
from functions import *


def fetch_volatile_coin(duration=30, sleep_time=600):
    stream = "wss://fstream.binance.com/ws/!ticker@arr"
    symbol_data = {}

    def get_volatile_dataframe():
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
        for _, row in frame.iterrows():
            symbol = row['s']
            symbol_data[symbol] = row.to_dict()

    def on_error(ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket connection closed.")

    def stop_ws(ws):
        print(f"Stopping websocket after {duration} seconds.")
        ws.close()

    ws = websocket.WebSocketApp(stream, on_message=on_message, on_error=on_error, on_close=on_close)
    file_name = "volatile_coins.csv"
    previous_coin = get_last_coin_from_csv(file_name)
    while True:
        try:
            timer = Timer(duration, stop_ws, [ws])
            timer.start()

            print('checking for new volatile coin')
            ws.run_forever()

            df = get_volatile_dataframe()
            volatile_coin = df.iloc[-1]['s']

            print(volatile_coin)

            volatile_coin = volatile_coin.split('USDT')[0]

            

            if previous_coin != volatile_coin:
                start_time = pd.Timestamp.now()  # Current time for new coin
                save_to_csv(volatile_coin, start_time)  # Save the new coin with its start_time
                send_mail("volatile_coins.csv")
                previous_coin = volatile_coin

            notifier(f'Shared_coin updated to {volatile_coin}')

            print(f'Sleeping for {sleep_time/60} minutes')
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            time.sleep(10)

def get_last_coin_from_csv(file_name):
    if os.path.exists(file_name):
        last_row = pd.read_csv(file_name)
        if not last_row.empty:
            return last_row['coin'].iloc[-1]
    return None


def send_mail(filename, subject='SARAVANA BHAVA'):
    from_ = 'gannamanenilakshmi1978@gmail.com'
    to = 'vamsikrishnagannamaneni@gmail.com'

    message = MIMEMultipart()
    message['From'] = from_
    message['To'] = to
    message['Subject'] = subject
    body_email = 'SARAVANA BHAVA !'

    message.attach(MIMEText(body_email, 'plain'))

    attachment = open(filename, 'rb')

    x = MIMEBase('application', 'octet-stream')
    x.set_payload((attachment).read())
    encoders.encode_base64(x)

    x.add_header('Content-Disposition', 'attachment; filename= %s' % filename)
    message.attach(x)

    s_e = smtplib.SMTP('smtp.gmail.com', 587)
    s_e.starttls()

    s_e.login(from_, 'upsprgwjgtxdbwki')
    text = message.as_string()
    s_e.sendmail(from_, to, text)
    print(f'Sent {filename}')

import pandas as pd
import os

def save_to_csv(coin, start_time, filename="volatile_coins.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["coin", "starttime"])

    new_data = pd.DataFrame({"coin": [coin], "starttime": [start_time]})
    
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(filename, index=False)


fetch_volatile_coin()