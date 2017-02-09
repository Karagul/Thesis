Backtest\_results\_anova
================
Jan
2017/2/9

The following data is a part of backtest resutls conducted from 2014/01/02 to 2016/07/01 with 13 cointegrated pairs. Model 1 is the conventilnal parametric algorithm (linear regression + Bollinger Bands, data: log returns) while Models 4-1 and 4-2 are nonparametric algorithms (RF regression + Stationary Bootstrap, data: raw prices(closing prices for historical data plus the price in 15:30 pm that day)).

``` r
backtest = read.csv("thesis_anova_data.csv")
head(backtest)
```

    ##   Pair.Y.X. coint_smaller X_std_smaller lookback thresh_param
    ## 1   EWA/EWC             1             0      100         0.75
    ## 2   EWA/EWC             1             0      100         1.00
    ## 3   EWA/EWC             1             0      100         1.50
    ## 4   EWA/EWC             1             0      200         0.75
    ## 5   EWA/EWC             1             0      200         1.00
    ## 6   EWA/EWC             1             0      200         1.50
    ##   thresh_nonparam mod1 mod4_1 mod4_2
    ## 1             90% 0.35   2.04   1.70
    ## 2             95% 0.96   2.02   1.80
    ## 3             99% 1.18   1.95   1.63
    ## 4             90% 0.31   1.33   1.33
    ## 5             95% 1.11   1.33   1.32
    ## 6             99% 1.06   1.32   1.20

The meaning of each column is as follows:

coint\_smaller: compared to the reversed pair (if Y:A and X:B then Y:B and X:A), this pair has lower p-value from a cointegration test.

X\_std\_smaller: regressor has lower volatility than the regressand. According to Vidyamurthy(2004), this is considered desireble condition

lookback: The length of lookback. To test each algorithms with arbitrary long lookback in case estimation of an optimal lookback length failed (which is not that rare) it is set to either 100,200 or 300 days.

thresh\_param: threshold of parametric algorithm (z-value).

thresh\_nonparam: threshold of nonparametric algorithm (confidence interval).

mod1 : Overall Sharpe ratio obtained from backtest with Model 1 with given parameters (lookback, thresh\_param).

mod 4\_1 :Overall Sharpe ratio obtained from backtest with Model 4-1 with given parameters (lookback,thresh\_nonparam).

mod 4\_2 :Overall Sharpe ratio obtained from backtest with Model 4-2 with given parameters (lookback,thresh\_nonparam).

``` r
summary(aov(mod1~coint_smaller*X_std_smaller, data=backtest))
```

    ##                Df Sum Sq Mean Sq F value Pr(>F)
    ## coint_smaller   1   0.05  0.0537   0.122  0.727
    ## X_std_smaller   1   0.38  0.3804   0.867  0.354
    ## Residuals     114  50.02  0.4388

For Model 1, both criteria has not significantly affected the resulting Sharpe ratio which means that it's rather situational.

``` r
summary(aov(mod4_1~coint_smaller*X_std_smaller, data=backtest))
```

    ##                Df Sum Sq Mean Sq F value Pr(>F)  
    ## coint_smaller   1   0.07  0.0724   0.260 0.6108  
    ## X_std_smaller   1   1.20  1.2012   4.323 0.0398 *
    ## Residuals     114  31.68  0.2779                 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
summary(aov(mod4_2~coint_smaller*X_std_smaller, data=backtest))
```

    ##                Df Sum Sq Mean Sq F value Pr(>F)  
    ## coint_smaller   1   0.18  0.1801   0.753 0.3874  
    ## X_std_smaller   1   1.40  1.4000   5.853 0.0171 *
    ## Residuals     114  27.27  0.2392                 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

For Models 4-1 and 4-2 on the other hand, the volatility of regressor compared to regressand has a statistically significant impact on overall Sharpe ratio. Therefore, results are more predictable if we are certain that one security in the pair has lower (or higher) volatility than the other and long-term cointegration can be assumed.

Interestingly, with these new algorithms, the coeffcient o f X\_std\_smaller is negative (that of Model 1 is positive). Therefore, the reverse of Vidyamurthy(2004)'s rule seems to be effective to the new nonparametric algorithms (the one with higher volatility should be the regressor).
