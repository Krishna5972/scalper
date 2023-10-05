import logging
import pandas as pd
import time
from datetime import datetime
logging.basicConfig(filename='trading_data_log.txt',  filemode='a',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import datetime,timedelta
from data_extraction import *


from functions import *

coin = 'API3'
timeframe = '15m'
# is_short_tradable(coin, timeframe)

# is_volatile_tradable(coin, timeframe)

is_long_tradable(coin, timeframe)