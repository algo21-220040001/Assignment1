# Assignment1

The code is divided into two files: one reproduces the factor from the reference and the other performs a stratification backtest on the factor.  

"因子复现.ipynb" calculates the quality factor from the references and derives z-scores for this factor for 2016-2020， with the underlying CSI 800 stock pool.  
"backtest_multi_V2.py" reads the factor value file, divides 800 stocks into five layers by factor value, and backtests them separately to get five return curves; calculates the information ratio between factor value and stock return from; the group with the highest long score and the group with the lowest short score, and backtests them to get the return curve of the long-short portfolio. Since this factor is a fundamental factor, the frequency of position adjustment is set at 10 days.
