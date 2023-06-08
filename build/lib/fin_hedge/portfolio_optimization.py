import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


# TO DESIGN



if __name__ == '__main__':
    # Define the tickers for the two stocks
    ticker1, ticker2 = 'AAPL', 'MSFT'
    # Download historical prices
    df = yf.download([ticker1, ticker2], '2018-01-01', '2023-01-01')
    df = df['Adj Close']
    # TO DESIGN