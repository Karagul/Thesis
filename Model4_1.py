#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code will only work on Quantopian Web IDE

Below line 144 is a copy of several packages that are not importable in Quantopian IDE
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
import scipy
from zipline.utils import tradingcalendar
import pytz
from sklearn.ensemble import RandomForestRegressor
from functools import partial

def initialize(context):
    # Quantopian backtester specific variables
    set_slippage(slippage.FixedSlippage(spread=0))
    #set_commission(commission.PerTrade(cost=1))
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.00))
    set_symbol_lookup_date('2014-01-01')
    context.Y = symbol('EWG')
    context.X = symbol('EWL')
    set_benchmark(context.Y)
    np.random.seed(1)
    context.lookback =100
    context.z_window = context.lookback
    context.ci_value = 0.05
    context.oob = np.array([])
    context.spread = np.array([])
    context.inLong = False
    context.inShort = False
    context.first = True

def handle_data(context, data):
   
    _Y_value = context.portfolio.positions[context.Y].amount * context.portfolio.positions[context.Y].last_sale_price
    _X_value = context.portfolio.positions[context.X].amount * context.portfolio.positions[context.X].last_sale_price
    _leverage = (abs(_Y_value) + abs(_X_value)) / context.portfolio.portfolio_value
    record(
            X_value = _X_value ,
            Y_value = _Y_value ,
            leverage = _leverage
    )
    
    if get_open_orders():
        return
    rf = RandomForestRegressor(n_estimators=100,oob_score=True)#,n_jobs=-1)
    if context.first:
        pricesY = data.history(context.Y,"price", 810,"1d").iloc[-(context.lookback*2)::]
        pricesX = data.history(context.X,"price", 810,"1d").iloc[-(context.lookback*2)::]
        for idx in range(context.lookback):
            Y_old = pricesY[idx:(context.lookback+idx)]
            X_old = pricesX[idx:(context.lookback+idx)]
            rf.fit(X_old[:,None],Y_old)
            context.spread = np.append(context.spread, Y_old[-1]-rf.predict(X_old[-1]))
            context.oob = np.append(context.oob, rf.oob_score_) 
        context.first = False
    now = get_datetime()
    exchange_time = now.astimezone(pytz.timezone('US/Eastern'))
    """
    if (exchange_time.year == 2014) and (exchange_time.month == 7) and (exchange_time.day == 1):
        context.lookback = 110
        context.z_window = context.lookback
        #print("if1")
    if (exchange_time.year == 2015) and (exchange_time.month == 1) and (exchange_time.day == 2):
        context.lookback = 40
        context.z_window = context.lookback
        #print("if2")
    if (exchange_time.year == 2015) and (exchange_time.month == 7) and (exchange_time.day == 1):
        context.lookback = 40
        context.z_window = context.lookback
        #print("if3")
    if (exchange_time.year == 2016) and (exchange_time.month == 1) and (exchange_time.day == 4):
        context.lookback = 40
        context.z_window = context.lookback
        #print("if4") 
    """  
    if not (exchange_time.hour == 15 and exchange_time.minute == 30):
        return
    
    pricesY = data.history(context.Y,"price", 410,"1d").iloc[-(context.lookback+1)::]
    pricesX = data.history(context.X,"price", 410,"1d").iloc[-(context.lookback+1)::]
    Y = pricesY[-(context.lookback)::]
    X = pricesX[-(context.lookback)::]
    rf = RandomForestRegressor(n_estimators=500,oob_score=True,n_jobs=-1)
    rf.fit(X[:,None],Y)
    result = rf.predict(X[-1])
    hedge = result/X[-1]
    context.spread = np.append(context.spread, Y[-1]-result)
    context.oob = np.append(context.oob, rf.oob_score_) 
    
    if context.spread.size > context.z_window:
        
        spreads = np.exp(-np.abs(context.spread[-context.z_window:]))
        CIs = ci(spreads, np.mean, n_samples=500,alpha=context.ci_value)
        if (context.inShort or context.inLong) and spreads[-1] > CIs[1]:
            order_target(context.Y, 0)
            order_target(context.X, 0)
            context.inShort = False
            context.inLong = False
            record(X_pct=0, Y_pct=0)
            return
        
        if spreads[-1] < CIs[0] and (not context.inLong) and context.spread[-1]<0:
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
        
        if spreads[-1] < CIs[0] and (not context.inShort)  and context.spread[-1]>0:#dif[-1]>0:#and context.spread[-1]>0 and dif[-1]>0:#and (not flag):
            # Only trade if NOT already in a trade
            y_target_shares = -1
            X_target_shares = hedge
            context.inShort = True
            context.inLong = False

           
            (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, Y[-1], X[-1] )
            order_target_percent(context.Y, y_target_pct)
            order_target_percent(context.X, x_target_pct)
            record(Y_pct=y_target_pct, X_pct=x_target_pct)

            return
    
def computeHoldingsPct(yShares, xShares, yPrice, xPrice):
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)


#Copyright (c) 2011, Josh Hemann  (hemann @ colorado . edu)
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the code's author, Josh Hemann, nor the
#      names of its contributors, may be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from collections import defaultdict

def resample(_data, p, seed=None):
    """
    Performs a stationary block bootstrap resampling of elements from a time 
    series, or rows of an input matrix representing a multivariate time series
    
    Inputs:
        data - An MxN numerical array of data to be resampled. It is assumed
               that rows correspond to measurement observations and columns to 
           items being measured. 1D arrays are allowed.
        p    - A scalar float in (0,1] defining the parameter for the geometric
               distribution used to draw block lengths. The expected block 
           length is then 1/p                

    Keywords:
        seed - A scalar integer defining the seed to use for the underlying
       random number generator.
    
    Return:
        A three element list containing
    - A resampled version, or "replicate" of data
    - A length M array of integers serving as indices into the rows
      of data that were resampled, where M is the number of rows in 
      data. Thus, if indices[i] = k then row i of the replicate data 
      contains row k of the original data.
    - A dictionary containing a mapping between indices into data and
      indices into the replicate data. Specifically, the keys are the
      indices for unique numbers in data and the associated dict values 
      are the indices into the replicate data where the numbers are.

    Example:            
        In [1]: import numpy as np
        In [2]: x = np.random.randint(0,20, 10)
        In [3]: import stationary_block_bootstrap as sbb
        In [4]: x_star, x_indices, x_indice_dict = sbb.resample(x, 0.333333)
        In [5]: x
        Out[5]: array([19,  2,  9,  9,  9, 10,  2,  2,  0, 11])
        In [6]: x_star
        Out[6]: array([19, 11,  2,  0, 11, 19,  2,  2, 19,  2])
        In [7]: x_indices
        Out[7]: array([0, 9, 7, 8, 9, 0, 6, 7, 0, 1])

        So, x_indices[1] = 9 means that the 1th element of x_star corresponds 
        to the 9th element of x
        
        In [8]: x_indice_dict
        Out[8]: {0: [0, 5, 8], 7: [2, 6, 7, 9], 8: [3], 9: [1, 4]}
    
        So, x[0] = 19 occurs in position 0, 5, and 8 in x_star. Likewise, 
        x[9] = 11 occurs in positions 1 and 4 in x_star
    
"""
    num_obs = np.shape(_data)[0]
    num_dims = np.ndim(_data)
    assert num_dims == 1 or num_dims == 2, "Input data must be a 1 or 2D array"
    #There is a probably smarter way to wrap the series without doubling
    #the data in memory; the approach below is easy but wasteful
    if num_dims == 1:
        wrapped_data = np.concatenate((_data, _data)) 
    elif num_dims == 2:
        wrapped_data = np.row_stack((_data, _data)) 
    
    assert p > 0 and p <=1, "p must be in (0,1]"
    
    if seed is not None:
        np.random.seed(seed=seed)

    #Make the random variables used in the resampling ahead of time. Could be
    #problematic for memory if num_obs is huge, but doing it this way cuts down 
    #on the function calls in the for loop below...
    choices = np.random.randint(0, num_obs, num_obs)
    unifs = np.random.uniform(0, 1, num_obs)
    
    #Let x and x* be length-n vectors with x*[0] = x[0]. 
    #Then x*[1] = x[1] with probability 1-p. With probability p, x*[1] will
    #equal a random i for x[i]. The expected run length is 1/p = "block length"
    indices = -np.ones(num_obs, dtype=int)
    indices[0] = 0
        
    for i in xrange(1, num_obs):
        if (unifs[i] > p): 
            indices[i] = indices[i-1] + 1 
        else:
            indices[i] = choices[i]

    if num_dims == 1:        
        resampled_data = wrapped_data[indices]   
        index_to_data_map = dict((x, i) for i, x in enumerate(wrapped_data))
        bootstrap_indices = map(index_to_data_map.get, resampled_data)
    elif num_dims == 2:
        #Mapping is the same for each column with respect to which rows are
        #resampled, so just consider one variable when mapping indices to data...
        resampled_data = wrapped_data[indices, :]   
        index_to_data_map = dict((x, i) for i, x in enumerate(wrapped_data[:,0]))
        bootstrap_indices = map(index_to_data_map.get, resampled_data[:,0])
        
    #bootstrap_indices = [index % num_obs for index in bootstrap_indices]
    
    #The code below is making a dictionary mapping of observations resampled to
    #where in the array of indices that observation shows up. Some observations
    #are resampled multiple times, others not at all, in any given replicate
    #data set. The straight-forward code is
    # try:
    #   items = dict[key]
    # except KeyError:
    #   dict[key] = items = [ ]
    #   items.append(value)
    """
    index_occurences = defaultdict(list)
    for pos, index in enumerate(bootstrap_indices):
        index_occurences[index].append(pos)
    
    index_dict = dict(index_occurences)
    """
    #Need to make the indices we save be bounded by 0:num_obs. For example, 
    #data[0,:] = data[num_obs,:]  and data[1,:] = data[num_obs+1,*] etc     
    #because we wrapped the data. But, with respect to the data arrays used 
    #elsewhere, an index of num_obs+1 is out of bounds, so num_obs should be 
    #converted to 0, num_obs+1 to 1, etc...   

    return [resampled_data, indices % num_obs]#, index_dict] 
    #end resample() 
    
"""    
if __name__ == "__main__":
    x = np.random.randint(0,20, 10)
    x_star, x_indices, x_indice_dict = resample(x, 0.333333) #expected block length = 3
    print 'Original series: ', x
    print 'Resampled series: ', x_star
    print 'Indices into original series: ', x_indices
    print 'Dictionary where key=index into original data, value=index into\n' \
          + '  resampled data where that value occurs: ''', x_indice_dict
    print '\n\n'

    y = np.arange(18).reshape((6,3))
    y_star, y_indices, y_indice_dict = resample(y, 0.5) #expected block length = 2
    print 'Original MxN series of M observations over N variables:\n', y
    print 'Resampled series:\n', y_star
    print 'Indices into rows of original series: ', y_indices
    print 'Dictionary where key=row in original data, value=rows in\n' \
          + '  resampled data where that observation occurs: ', y_indice_dict
"""
from scipy.stats import norm
def ci(data, statfunction=np.average, alpha=0.05, n_samples=10000, method='bca', output='lowhigh', epsilon=0.001, multi=None):
    """
Given a set of data ``data``, and a statistics function ``statfunction`` that
applies to that data, computes the bootstrap confidence interval for
``statfunction`` on that data. Data points are assumed to be delineated by
axis 0.
Parameters
----------
data: array_like, shape (N, ...) OR tuple of array_like all with shape (N, ...)
    Input data. Data points are assumed to be delineated by axis 0. Beyond this,
    the shape doesn't matter, so long as ``statfunction`` can be applied to the
    array. If a tuple of array_likes is passed, then samples from each array (along
    axis 0) are passed in order as separate parameters to the statfunction. The
    type of data (single array or tuple of arrays) can be explicitly specified
    by the multi parameter.
statfunction: function (data, weights=(weights, optional)) -> value
    This function should accept samples of data from ``data``. It is applied
    to these samples individually. 
    
    If using the ABC method, the function _must_ accept a named ``weights`` 
    parameter which will be an array_like with weights for each sample, and 
    must return a _weighted_ result. Otherwise this parameter is not used
    or required. Note that numpy's np.average accepts this. (default=np.average)
alpha: float or iterable, optional
    The percentiles to use for the confidence interval (default=0.05). If this
    is a float, the returned values are (alpha/2, 1-alpha/2) percentile confidence
    intervals. If it is an iterable, alpha is assumed to be an iterable of
    each desired percentile.
n_samples: float, optional
    The number of bootstrap samples to use (default=10000)
method: string, optional
    The method to use: one of 'pi', 'bca', or 'abc' (default='bca')
output: string, optional
    The format of the output. 'lowhigh' gives low and high confidence interval
    values. 'errorbar' gives transposed abs(value-confidence interval value) values
    that are suitable for use with matplotlib's errorbar function. (default='lowhigh')
epsilon: float, optional (only for ABC method)
    The step size for finite difference calculations in the ABC method. Ignored for
    all other methods. (default=0.001)
multi: boolean, optional
    If False, assume data is a single array. If True, assume data is a tuple/other
    iterable of arrays of the same length that should be sampled together. If None,
    decide based on whether the data is an actual tuple. (default=None)
    
Returns
-------
confidences: tuple of floats
    The confidence percentiles specified by alpha
Calculation Methods
-------------------
'pi': Percentile Interval (Efron 13.3)
    The percentile interval method simply returns the 100*alphath bootstrap
    sample's values for the statistic. This is an extremely simple method of 
    confidence interval calculation. However, it has several disadvantages 
    compared to the bias-corrected accelerated method, which is the default.
'bca': Bias-Corrected Accelerated (BCa) Non-Parametric (Efron 14.3) (default)
    This method is much more complex to explain. However, it gives considerably
    better results, and is generally recommended for normal situations. Note
    that in cases where the statistic is smooth, and can be expressed with
    weights, the ABC method will give approximated results much, much faster.
    Note that in a case where the statfunction results in equal output for every
    bootstrap sample, the BCa confidence interval is technically undefined, as
    the acceleration value is undefined. To match the percentile interval method
    and give reasonable output, the implementation of this method returns a
    confidence interval of zero width using the 0th bootstrap sample in this
    case, and warns the user.  
'abc': Approximate Bootstrap Confidence (Efron 14.4, 22.6)
    This method provides approximated bootstrap confidence intervals without
    actually taking bootstrap samples. This requires that the statistic be 
    smooth, and allow for weighting of individual points with a weights=
    parameter (note that np.average allows this). This is _much_ faster
    than all other methods for situations where it can be used.
Examples
--------
To calculate the confidence intervals for the mean of some numbers:
>> boot.ci( np.randn(100), np.average )
Given some data points in arrays x and y calculate the confidence intervals
for all linear regression coefficients simultaneously:
>> boot.ci( (x,y), scipy.stats.linregress )
References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
    """

    # Deal with the alpha values
    if np.iterable(alpha):
        alphas = np.array(alpha)
    else:
        alphas = np.array([alpha/2,1-alpha/2])

    if multi == None:
      if isinstance(data, tuple):
        multi = True
      else:
        multi = False

    # Ensure that the data is actually an array. This isn't nice to pandas,
    # but pandas seems much much slower and the indexes become a problem.
    if multi == False:
      data = np.array(data)
      tdata = (data,)
    else:
      tdata = tuple( np.array(x) for x in data )

    # Deal with ABC *now*, as it doesn't need samples.
    if method == 'abc':
        n = tdata[0].shape[0]*1.0
        nn = tdata[0].shape[0]

        I = np.identity(nn)
        ep = epsilon / n*1.0
        p0 = np.repeat(1.0/n,nn)

        t1 = np.zeros(nn); t2 = np.zeros(nn)
        try:
          t0 = statfunction(*tdata,weights=p0)
        except TypeError as e:
          raise TypeError("statfunction does not accept correct arguments for ABC ({0})".format(e.message))

        # There MUST be a better way to do this!
        for i in range(0,nn):
            di = I[i] - p0
            tp = statfunction(*tdata,weights=p0+ep*di)
            tm = statfunction(*tdata,weights=p0-ep*di)
            t1[i] = (tp-tm)/(2*ep)
            t2[i] = (tp-2*t0+tm)/ep**2

        sighat = np.sqrt(np.sum(t1**2))/n
        a = (np.sum(t1**3))/(6*n**3*sighat**3)
        delta = t1/(n**2*sighat)
        cq = (statfunction(*tdata,weights=p0+ep*delta)-2*t0+statfunction(*tdata,weights=p0-ep*delta))/(2*sighat*ep**2)
        bhat = np.sum(t2)/(2*n**2)
        curv = bhat/sighat-cq
        z0 = norm.ppf(2*norm.cdf(a)*norm.cdf(-curv))
        Z = z0+norm.ppf(alphas)
        za = Z/(1-a*Z)**2
        # stan = t0 + sighat * norm.ppf(alphas)
        abc = np.zeros_like(alphas)
        for i in range(0,len(alphas)):
            abc[i] = statfunction(*tdata,weights=p0+za[i]*delta)

        if output == 'lowhigh':
            return abc
        elif output == 'errorbar':
            return abs(abc-statfunction(tdata))[np.newaxis].T
        else:
            raise ValueError("Output option {0} is not supported.".format(output))

    # We don't need to generate actual samples; that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    bootindexes = bootstrap_indexes( tdata[0], n_samples )
    stat = np.array([statfunction(*(x[indexes] for x in tdata)) for indexes in bootindexes])
    stat.sort(axis=0)

    # Percentile Interval Method
    if method == 'pi':
        avals = alphas

    # Bias-Corrected Accelerated Method
    elif method == 'bca':

        # The value of the statistic function applied just to the actual data.
        ostat = statfunction(*tdata)

        # The bias correction value.
        z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

        # Statistics of the jackknife distribution
        jackindexes = jackknife_indexes(tdata[0])
        jstat = [statfunction(*(x[indexes] for x in tdata)) for indexes in jackindexes]
        jmean = np.mean(jstat,axis=0)

        # Acceleration value
        a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )
        if np.any(np.isnan(a)):
            nanind = np.nonzero(np.isnan(a))
    
        zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

        avals = norm.cdf(z0 + zs/(1-a*zs))

    else:
        raise ValueError("Method {0} is not supported.".format(method))

    nvals = np.round((n_samples-1)*avals)
    


    nvals = np.nan_to_num(nvals).astype('int')

    if output == 'lowhigh':
        if nvals.ndim == 1:
            # All nvals are the same. Simple broadcasting
            return stat[nvals]
        else:
            # Nvals are different for each data point. Not simple broadcasting.
            # Each set of nvals along axis 0 corresponds to the data at the same
            # point in other axes.
            return stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]
    elif output == 'errorbar':
        if nvals.ndim == 1:
          return abs(statfunction(data)-stat[nvals])[np.newaxis].T
        else:
          return abs(statfunction(data)-stat[(nvals, np.indices(nvals.shape)[1:])])[np.newaxis].T
    else:
        raise ValueError("Output option {0} is not supported.".format(output))
    
    



def ci_abc(data, stat=lambda x,y: np.average(x,weights=y) , alpha=0.05, epsilon = 0.001):
    """
.. note:: Deprecated. This functionality is now rolled into ci.
          
Given a set of data ``data``, and a statistics function ``statfunction`` that
applies to that data, computes the non-parametric approximate bootstrap
confidence (ABC) interval for ``stat`` on that data. Data points are assumed
to be delineated by axis 0.
Parameters
----------
data: array_like, shape (N, ...)
    Input data. Data points are assumed to be delineated by axis 0. Beyond this,
    the shape doesn't matter, so long as ``statfunction`` can be applied to the
    array.
stat: function (data, weights) -> value
    The _weighted_ statistic function. This must accept weights, unlike for other
    methods.
alpha: float or iterable, optional
    The percentiles to use for the confidence interval (default=0.05). If this
    is a float, the returned values are (alpha/2, 1-alpha/2) percentile confidence
    intervals. If it is an iterable, alpha is assumed to be an iterable of
    each desired percentile.
epsilon: float
    The step size for finite difference calculations. (default=0.001)
Returns
-------
confidences: tuple of floats
    The confidence percentiles specified by alpha
References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
bootstrap R package: http://cran.r-project.org/web/packages/bootstrap/
    """
    return ci(data, statfunction=lambda x,weights: stat(x,weights), alpha=alpha, epsilon=epsilon,
            method='abc')

def bootstrap_indexes(data, n_samples=10000):
    """
Given data points data, where axis 0 is considered to delineate points, return
an generator for sets of bootstrap indexes. This can be used as a list
of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
    """
    for _ in xrange(n_samples):
        yield resample(data,p=0.333333)[1]#0.333333

def jackknife_indexes(data):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of jackknife indexes.
For a given set of data Y, the jackknife sample J[i] is defined as the data set
Y with the ith data point deleted.
    """
    base = np.arange(0,len(data))
    return (np.delete(base,i) for i in base)

def subsample_indexes(data, n_samples=1000, size=0.5):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is indexes a subsample of the data of size
``size``. If size is >= 1, then it will be taken to be an absolute size. If
size < 1, it will be taken to be a fraction of the data size. If size == -1, it
will be taken to mean subsamples the same size as the sample (ie, permuted
samples)
    """
    if size == -1:
        size = len(data)
    elif (size < 1) and (size > 0):
        size = round(size*len(data))
    elif size > 1:
        pass
    else:
        raise ValueError("size cannot be {0}".format(size))
    base = np.tile(np.arange(len(data)),(n_samples,1))
    for sample in base: np.random.shuffle(sample)
    return base[:,0:size]
  