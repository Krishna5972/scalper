import pandas as pd
from functions import notifier

stake = 600
timeframe = '5m'
initial_stake = 600 #for ploting

#short term
short_term_period = 1
short_term_atr_multiplier = 1
short_term_pivot_period = 1

#long_term

 #           pivot_st = PivotSuperTrendConfiguration(period = trade_config.short_term_period, atr_multiplier = trade_config.short_term_atr_multiplier, pivot_period = trade_config.short_term_pivot_period)
long_term_period = 3  #2
long_term_atr_multiplier = 3 #2.8
long_term_pivot_period = 3  #2


df = pd.read_csv('trades.csv')
df['utc_time'] = pd.to_datetime(df['utc_time'])
df['year'] = df['utc_time'].dt.year
df['month'] = df['utc_time'].dt.month

df_PNL = df.groupby(['year','month','date']).agg({'realizedPnl' : 'sum'}).reset_index()

yesterdays_PNL = df_PNL.iloc[-1]['realizedPnl']

if yesterdays_PNL < 0:
    stake_multipler = 1.32
else:
    stake_multipler = 1

notifier(f"Stake multiplier is set to {stake_multipler} as yesterday returns were {round(yesterdays_PNL/initial_stake,4)*100}")