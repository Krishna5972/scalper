from functions import *

class TradeConfiguration:
    def __init__(self):
        self.buy_risk = 0.0123  #0.0213
        self.sell_risk = 0.006  #0.0123
        
        
    def get_risk(self,over_all_trend,current_signal):
        if over_all_trend  == "Uptrend":
            if current_signal == 'Buy':  
                risk = self.buy_risk
            else:
                risk = self.sell_risk/2
        else:
            if current_signal == 'Buy':
                risk = self.buy_risk/1.5
            else:
                risk = self.sell_risk
            
        #notifier(f'Overall Trend is {over_all_trend} and Current Trend is {current_signal}, so risk is set to {risk}')
        return risk   
    
class PivotSuperTrendConfiguration():
    def __init__(self,period = 12, atr_multiplier = 1, pivot_period = 10):
        self.period = period
        self.atr_multiplier = atr_multiplier
        self.pivot_period = pivot_period
        
        
        
        
class Order:
    def __init__(self,coin,entry,quantity,round_price,take_profit = None , stop_loss = None):
        self.coin = coin.upper()
        self.quantity = quantity
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.entry = entry
        self.round_price = round_price
        
    def make_buy_trade(self,client):
        
        
        
        client.futures_create_order(symbol=f'{self.coin}USDT', side='BUY', type='MARKET', quantity=self.quantity, dualSidePosition=True, positionSide='LONG')
             
        self.take_profit = self.entry+((self.entry*0.06))
        
        client.futures_create_order(
                                    symbol=f'{self.coin}USDT',
                                    price=round(self.take_profit, self.round_price),
                                    side='SELL',
                                    positionSide='LONG',
                                    quantity=self.quantity,
                                    timeInForce='GTC',
                                    type='LIMIT',
                                    # reduceOnly=True,
                                    closePosition=False,
                                    # stopPrice=round(take_profit,2),
                                    workingType='MARK_PRICE',
                                    priceProtect=True
                                )
        notifier(f'Coin :{self.coin}, Quantity : {self.quantity } stake : {round(self.quantity*self.entry,2)}')
        notifier(f'Buy order placed for coin :{self.coin}, TP : {self.take_profit}')
        
    def make_sell_trade(self,client):
        
        
        client.futures_create_order(
                                        symbol=f'{self.coin}USDT', side='SELL', 
                                        type='MARKET',
                                        quantity=self.quantity,
                                        dualSidePosition=True, 
                                        positionSide='SHORT'
                                    )
        
        self.take_profit = self.entry - ((self.entry * 0.0411))
        
        client.futures_create_order(
                                    symbol=f'{self.coin}USDT',
                                    price=round(self.take_profit, self.round_price),
                                    side='BUY',
                                    positionSide='SHORT',
                                    quantity=self.quantity,
                                    timeInForce='GTC',
                                    type='LIMIT',
                                    # reduceOnly=True,
                                    closePosition=False,
                                    # stopPrice=round(take_profit,2),
                                    workingType='MARK_PRICE',
                                    priceProtect=True
                               )
        
        notifier(f'Coin :{self.coin}, Quantity : {self.quantity } stake : {round(self.quantity*self.entry,2)}')
        notifier(f'Sell order placed for coin :{self.coin}, TP : {self.take_profit}')
        
    