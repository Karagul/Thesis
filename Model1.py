#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code will only work on Quantopian Web IDE

"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
from zipline.utils import tradingcalendar
import pytz


def initialize(context):
    # Quantopian backtester specific variables
    set_slippage(slippage.FixedSlippage(spread=0))
    #set_commission(commission.PerTrade(cost=1))
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.00))
    set_symbol_lookup_date('2014-01-01')
    context.Y = symbol('EWG')
    context.X = symbol('EWL')
    set_benchmark(context.Y)
    
    
    # strategy specific variables
    context.lookback = 200 
    context.z_window = context.lookback
    
    context.useHRlag = False
    context.HRlag = 1
    
    context.spread = np.array([])
    context.hedgeRatioTS = np.array([])
    context.hedgeRatioTS_ = np.array([])
    context.inLong = False
    context.inShort = False
    context.entryZ = 1.00
    context.exitZ = 0.0
    context.first = True
    if not context.useHRlag:
        # a lag of 1 means no-lag, this is used for np.array[-1] indexing
        context.HRlag = 1
        
# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    _Y_value = context.portfolio.positions[context.Y].amount * context.portfolio.positions[context.Y].last_sale_price
    _X_value = context.portfolio.positions[context.X].amount * context.portfolio.positions[context.X].last_sale_price
    _leverage = (abs(_Y_value) + abs(_X_value)) / context.portfolio.portfolio_value
    record(
            X_value = _X_value ,
            Y_value = _Y_value ,
            leverage = _leverage,
            
            
    ) 
    
    if context.first:
        pricesY = data.history(context.Y,"price", 610,"1d").iloc[-(context.lookback*2+1)::]
        pricesX = data.history(context.X,"price", 610,"1d").iloc[-(context.lookback*2+1)::]        
        
        for idx in range(context.lookback):
            Y_old = pricesY[idx:(context.lookback+idx)]
            X_old = pricesX[idx:(context.lookback+idx)]
            hedge = hedge_ratio(Y_old, X_old, add_const=True)
            context.hedgeRatioTS = np.append(context.hedgeRatioTS, hedge)
            Y_old = pricesY[idx:(context.lookback+idx+1)]
            X_old = pricesX[idx:(context.lookback+idx+1)] 
            Y_returns = np.log(Y_old/Y_old.shift(1))[1:(context.lookback+idx+1)]
            X_returns = np.log(X_old/X_old.shift(1))[1:(context.lookback+idx+1)]
            hedge_ = hedge_ratio(Y_returns, X_returns, add_const=True)
            context.hedgeRatioTS_ = np.append(context.hedgeRatioTS_, hedge_)
            if context.hedgeRatioTS.size < context.HRlag:
                return
            hedge = context.hedgeRatioTS[-context.HRlag] 
            hedge_ = context.hedgeRatioTS_[-context.HRlag]  
            context.spread = np.append(context.spread, Y_returns[-1] - hedge_ * X_returns[-1])  
        context.first = False
  
    if get_open_orders():
        return
    
    now = get_datetime()
    exchange_time = now.astimezone(pytz.timezone('US/Eastern'))
    """   
    if (exchange_time.year == 2014) and (exchange_time.month == 7) and (exchange_time.day == 1):
        context.lookback = 150
        context.z_window = context.lookback
        #print("if1")
    if (exchange_time.year == 2015) and (exchange_time.month == 1) and (exchange_time.day == 2):
        context.lookback = 50
        context.z_window = context.lookback
        #print("if2")
    if (exchange_time.year == 2015) and (exchange_time.month == 7) and (exchange_time.day == 1):
        context.lookback = 50
        context.z_window = context.lookback
        #print("if3")
    if (exchange_time.year == 2016) and (exchange_time.month == 1) and (exchange_time.day == 4):
        context.lookback = 50
        context.z_window = context.lookback
        #print("if4") 
    """    
    if not (exchange_time.hour == 15 and exchange_time.minute == 30):
        return
    
    pricesY = data.history(context.Y,"price", 320,"1d").iloc[-(context.lookback+1)::]
    pricesX = data.history(context.X,"price", 320,"1d").iloc[-(context.lookback+1)::]
    print([context.lookback,context.entryZ])
    Y = pricesY[-(context.lookback)::]
    X = pricesX[-(context.lookback)::]
    X_returns = np.log(X/pricesX.shift(1))[-(context.lookback)::]
    Y_returns = np.log(Y/pricesY.shift(1))[-(context.lookback)::]

    try:
        hedge = hedge_ratio(Y, X, add_const=True)      
    except ValueError as e:
        log.debug(e)
        return
    
    context.hedgeRatioTS = np.append(context.hedgeRatioTS, hedge)
    hedge_ = hedge_ratio(Y_returns,X_returns, add_const=True)
    context.hedgeRatioTS_ = np.append(context.hedgeRatioTS_, hedge_)    
    # Calculate the current day's spread and add it to the running tally
    if context.hedgeRatioTS.size < context.HRlag:
        return
    # Grab the previous day's hedgeRatio
    hedge = context.hedgeRatioTS[-context.HRlag] 
    hedge_ = context.hedgeRatioTS_[-context.HRlag]  
    context.spread = np.append(context.spread, Y_returns[-1] - hedge_ * X_returns[-1])

    if context.spread.size > context.z_window:
        # Keep only the z-score lookback period
        spreads = context.spread[-context.z_window:]
        
        zscore = (spreads[-1] - spreads.mean()) / spreads.std()
        #print([spreads.mean(),spreads.std(),context.lookback])
        
        if context.inShort and zscore < context.exitZ:
            order_target(context.Y, 0)
            order_target(context.X, 0)
            context.inShort = False
            context.inLong = False
            record(X_pct=0, Y_pct=0)
            return
        
        if context.inLong and zscore > context.exitZ:
            order_target(context.Y, 0)
            order_target(context.X, 0)
            context.inShort = False
            context.inLong = False
            record(X_pct=0, Y_pct=0)
            return
            
        if zscore < -context.entryZ and (not context.inLong):
            # Only trade if NOT already in a trade
            y_target_shares = 1
            X_target_shares = -hedge
            context.inLong = True
            context.inShort = False
            
            (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares,X_target_shares, Y[-1], X[-1] )
            order_target_percent(context.Y, y_target_pct)
            order_target_percent(context.X, x_target_pct)
            record(Y_pct=y_target_pct, X_pct=x_target_pct)
            return

        if zscore > context.entryZ and (not context.inShort):
            # Only trade if NOT already in a trade
            y_target_shares = -1
            X_target_shares = hedge
            context.inShort = True
            context.inLong = False
           
            (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, Y[-1], X[-1] )
            order_target_percent(context.Y, y_target_pct)
            order_target_percent(context.X, x_target_pct)
            record(Y_pct=y_target_pct, X_pct=x_target_pct)

def is_market_close(dt):
    ref = tradingcalendar.canonicalize_datetime(dt)
    return dt == tradingcalendar.open_and_closes.T[ref]['market_close']

def hedge_ratio(Y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        return model.params[1]
    model = sm.OLS(Y, X).fit()
    return model.params.values
    
def computeHoldingsPct(yShares, xShares, yPrice, xPrice):
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)


