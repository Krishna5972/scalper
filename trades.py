from binance.client import Client
from datetime import datetime,timedelta
import pandas as pd
from scipy.stats import mode
import config
from functions import *

master_df = pd.read_csv('trades.csv')

def convert_timestamp_to_utc(timestamp_in_milliseconds):
    timestamp_in_seconds = timestamp_in_milliseconds / 1000.0
    return datetime.utcfromtimestamp(timestamp_in_seconds)

client=Client(config.api_key,config.secret_key)

account_history = client.futures_account_trades(limit = 1000)
df = pd.DataFrame(account_history)

df['utc_time'] = df['time'].apply(convert_timestamp_to_utc)
df['date'] = df['utc_time'].dt.day
df['minute'] = df['utc_time'].dt.minute
df['realizedPnl'] = df['realizedPnl'].astype(float)


aggregations = {
    'symbol': lambda x: x.mode().iloc[0] if x.value_counts().iloc[0] > 1 else x.iloc[0],
    'realizedPnl': 'sum', 
    'utc_time': lambda x: mode(x).mode[0],
    'date' : lambda x: mode(x).mode[0],
    'minute' : lambda x: mode(x).mode[0],
    'positionSide' : lambda x: x.mode().iloc[0] if x.value_counts().iloc[0] > 1 else x.iloc[0],
    'maker' : lambda x: x.mode().iloc[0] if x.value_counts().iloc[0] > 1 else x.iloc[0],
}

PNL = df.groupby('orderId').agg(aggregations).reset_index()
PNL = PNL.sort_values(by = 'utc_time').reset_index(drop=True)
PNL['utc_time'] = PNL['utc_time'].shift(1)
PNL = PNL[PNL['realizedPnl']!=0]
PNL = PNL.sort_values(by = 'utc_time').reset_index(drop=True)

PNL = pd.concat([master_df,PNL],axis = 0)
PNL = PNL.drop_duplicates(subset=['orderId','symbol'], keep='last')

PNL.to_csv(f'trades.csv',index=False,mode='w+')

send_mail('trades.csv')