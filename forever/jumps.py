import json
import websocket
from threading import Timer
import sys
sys.path.append('..')
from functions import *

MAX_SNAPSHOTS = 10
previous_snapshots = []
start_jump_times = {}


MAX_SNAPSHOTS = 10
previous_snapshots = [] 
symbol_data = {}

def get_rank_difference_dataframe():
    current_df = pd.DataFrame.from_dict(symbol_data, orient='index')
    current_df['CurrentRank'] = current_df['Volatility'].rank(ascending=False)

    rank_difference_data = {
        'coin': [],
    }
    for i in range(MAX_SNAPSHOTS):
        rank_difference_data[f'difference{i + 1}'] = []

    for coin, row in current_df.iterrows():
        rank_difference_data['coin'].append(coin)
        for i, snapshot in enumerate(previous_snapshots):
            prev_rank = snapshot.at[coin, 'Rank'] if coin in snapshot.index else None
            curr_rank = row['CurrentRank']
            diff = prev_rank - curr_rank if prev_rank is not None else 0
            rank_difference_data[f'difference{i + 1}'].append(diff)

    rank_difference_df = pd.DataFrame(rank_difference_data)
    rank_difference_df['TotalDifference'] = rank_difference_df.iloc[:, 1:].sum(axis=1)
    return rank_difference_df.sort_values(by='TotalDifference', ascending=False)


def on_message(ws, message,duration=30, sleep_time=6):
    global previous_snapshots
    msg = json.loads(message)
    symbols = [x for x in msg if x['s'].endswith('USDT')]
    frame = pd.DataFrame(symbols)[['s', 'o', 'h', 'l', 'c']]
    frame[['o', 'h', 'l', 'c']] = frame[['o', 'h', 'l', 'c']].astype(float)
    frame['Volatility'] = (frame['h'] - frame['l']) / abs(frame['l']) * 100
    frame['Rank'] = frame['Volatility'].rank(ascending=False)
    frame.set_index('s', inplace=True)

    previous_snapshots.append(frame)
    if len(previous_snapshots) > MAX_SNAPSHOTS:
        previous_snapshots.pop(0)

    if len(previous_snapshots) == MAX_SNAPSHOTS:
        df = get_rank_difference_dataframe()
        most_volatile_coin = df.iloc[0]['coin']
        print(f'Most volatile coin based on rank differences: {most_volatile_coin}')


    stream = "wss://fstream.binance.com/ws/!ticker@arr"
    symbol_data = {}

    def get_volatile_dataframe():
        df = pd.DataFrame.from_dict(symbol_data, orient='index')
        df['PctRange'] = (df['h'] - df['l']) / abs(df['l']) * 100
        df['Volatility'] = df['PctRange'].rolling(window=1).mean()
        
        df['Rank'] = df['Volatility'].rank(ascending=False)

        # Calculate rank jump for each of the previous snapshots
        if previous_snapshots:
            for i, snapshot in enumerate(previous_snapshots):
                df = df.merge(snapshot, on='s', suffixes=('', f'_previous_{i}'))
                df[f'RankJump_{i}'] = df[f'Rank_previous_{i}'] - df['Rank']

                # Check if a coin made a considerable jump into the top 20 ranks
                for _, row in df.iterrows():
                    symbol = row['s']
                    if row['Rank'] <= 20 and row[f'RankJump_{i}'] > 0 and row['Volatility'] > 5:
                        if symbol not in start_jump_times:
                            start_jump_times[symbol] = symbol_data[symbol]['E']

        # Calculate the average rank jump
        if len(previous_snapshots) > 0:
            rank_jump_cols = [f'RankJump_{i}' for i in range(len(previous_snapshots))]
            df['AverageRankJump'] = df[rank_jump_cols].mean(axis=1)
        else:
            df['AverageRankJump'] = 0
        
        df_volatility = df.sort_values(by='Volatility')
        return df_volatility


    def on_error(ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket connection closed.")

    def stop_ws(ws):
        print(f"Stopping websocket after {duration} seconds.")
        ws.close()

    ws = websocket.WebSocketApp(stream, on_message=on_message, on_error=on_error, on_close=on_close)
    file_name = "volatile_coins_jumps.csv"
    previous_coin = get_last_coin_from_csv(file_name)
    while True:
        try:
            timer = Timer(duration, stop_ws, [ws])
            timer.start()

            print('checking for new volatile coin')
            ws.run_forever()

            df = get_volatile_dataframe()
            volatile_coin = df.iloc[-1]['s']

            current_snapshot = df[['s', 'Rank']]
            previous_snapshots.append(current_snapshot)
            if len(previous_snapshots) > MAX_SNAPSHOTS:
                previous_snapshots.pop(0)
            df = df.sort_values(by='AverageRankJump', ascending=False)
            volatile_coin = df.iloc[0]['s']

            if previous_coin != volatile_coin:
                start_time = pd.Timestamp.now()  # Current time for new coin
                jump_start_time = start_jump_times.get(volatile_coin, pd.Timestamp.now())
                save_to_csv_jumps(volatile_coin, start_time, jump_start_time)
                send_mail("volatile_coins_jumps.csv")
                previous_coin = volatile_coin

            print(f'Shared_coin updated to {volatile_coin}')

            print(f'Sleeping for {sleep_time/60} minutes')
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
def save_to_csv_jumps(coin, start_time, jump_start_time, filename="volatile_coins_jumps.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["coin", "starttime", "jumpstart"])

    new_data = pd.DataFrame({"coin": [coin], "starttime": [start_time], "jumpstart": [jump_start_time]})
    
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(filename, index=False)


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


get_rank_difference_dataframe()