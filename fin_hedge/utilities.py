import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

""" Cointegration Test """
def cointegration_test(df, ticker1, ticker2):
    # Check if the pair is cointegrated
    score, pvalue, _ = coint(df[ticker1], df[ticker2])
    print(f'p-value: {pvalue}')
    # If p-value is small, we can assume the pair is cointegrated
    if pvalue < 0.05:
        print(f'{ticker1} and {ticker2} are cointegrated')
        # Calculate the hedge ratio
        model = sm.OLS(df[ticker1], df[ticker2])
        result = model.fit()
        hedge_ratio = result.params[ticker2]
        print(f'Hedge Ratio: {hedge_ratio}')
        # Create spread
        spread = df[ticker1] - hedge_ratio * df[ticker2]
        # Calculate z-score of the spread
        z_score = (spread - spread.mean()) / spread.std()
        # Create signals
        entry_threshold = 1.0  # Enter trade if z-score is > 1 or < -1
        exit_threshold = 0.0   # Exit trade if z-score is 0
        signals = pd.DataFrame(index=df.index)
        signals['longs'] = (z_score < -entry_threshold)  # Long the spread when z-score is below -1
        signals['shorts'] = (z_score > entry_threshold)  # Short the spread when z-score is above 1
        signals['exits'] = (np.abs(z_score) <= exit_threshold)  # Exit when z-score approaches 0
        signals['positions'] = signals['longs'] - signals['shorts']  # Create positions
        print(signals)
    else:
        print(f'{ticker1} and {ticker2} are not cointegrated')

""" ADF test """
def adf_test(df, ticker):
    # We'll use the closing price for our analysis
    price = df[ticker]
    # Conduct the ADF test
    result = adfuller(price)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    # A dictionary with the critical values for the test statistic at the 1, 5, and 10 percent levels
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))




if __name__ == '__main__':
    # Download historical prices
    ticker1, ticker2 = 'AAPL', 'MSFT'
    df = yf.download([ticker1, ticker2], '2018-01-01', '2023-01-01')
    df = df['Adj Close']
    # test functions
    cointegration_test(df, ticker1, ticker2)
    adf_test(df, ticker1)