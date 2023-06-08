import numpy as np
import pandas as pd
import yfinance as yf

""" Discounted Cash Flow (General)"""
def dcf(cash_flows, discount_rate):
    value = sum(cash_flow / (1 + discount_rate)**i for i, cash_flow in enumerate(cash_flows))
    return value

""" PE Ratio """
def pe_ratio(price, earnings):
    pe_ratio = price / earnings
    return pe_ratio

""" Dividend Discount Model """
def ddm(dividends, discount_rate):
    value = sum(dividend / (1 + discount_rate)**i for i, dividend in enumerate(dividends))
    return value

""" Discounted Cash Flow (for bonds) """
def dcf_bond(face_value, coupon_rate, payments_per_year, years_to_maturity, required_yield):
    coupon_payment = face_value * coupon_rate / payments_per_year
    price = 0.0
    for i in range(1, int(years_to_maturity * payments_per_year) + 1):
        price += coupon_payment / (1 + required_yield / payments_per_year) ** i
    price += face_value / (1 + required_yield / payments_per_year) ** (years_to_maturity * payments_per_year)
    print(f"The price of the bond is: ${round(bond_price, 2)}")
    return price

""" Black-Scholes """
def euro_vanilla_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    return call








""" Test Functions with dummy data """
if __name__ == '__main__':

    # DCF - general
    projected_cash_flows = [50, 55, 60, 65, 70]
    discount_rate = 0.10
    dcf_value = dcf(projected_cash_flows, discount_rate)

    # DCF - bonds
    face_value = 1000  # The bond's face value is $1000
    coupon_rate = 0.05  # The annual coupon rate is 5%
    payments_per_year = 2  # The bond pays coupon semi-annually
    years_to_maturity = 5  # The bond matures in 5 years
    required_yield = 0.04  # The required yield is 4%
    bond_price = dcf_bond(face_value, coupon_rate, payments_per_year, years_to_maturity, required_yield)

    # Euro Vanilla Option - Black-Scholes
    ticker = 'AAPL'
    data = yf.Ticker(ticker)
    spot_price = data.info['regularMarketPrice']
    dividend_rate = data.info['dividendRate'] / spot_price
    strike_price = 150.0
    volatility = 0.25  # Historical volatility, can be estimated based on historical prices
    interest_rate = 0.01  # Risk-free rate, e.g., current Treasury yield
    expiration_time = 1 / 12  # Time to expiration in years (1/12 represents one month)
    option_price = euro_vanilla_call(spot_price, strike_price, expiration_time, interest_rate, dividend_rate,
                                     volatility)
    print('The price of the call option is: $', round(option_price, 2))