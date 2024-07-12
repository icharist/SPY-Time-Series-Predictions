'''
SPY ETF Analysis
'''
# # Analysis of "The Market"
# 
# Typically whenever someone talks about beating "the market" they are talking about an index. The most common indices you may hear mentioned are the S&P500, NASDAQ, and the Dow Jones Industrial. These indices are a basket of stocks created upon different definitions. Those indices can be bought through exchange traded funds (ETFs) are known by their tickers "SPY", "QQQ", and "DIA". Since these are ETFs and not the true indices themselves they may differ slightly from the actual index they are mimicking. This analysis will focus on the SPY and DIA as they are more commonly know.
# 
# - SPY: The SPDR S&P 500 ETF is the largest ETF tracking the S&P 500. Coincidentally, as the name implies, the S&P500 is comprised of 500 companies. The index is constructed using a weighted average market capitalization, which means larger companies have a greater weighting in the index. 
# 
# 
# - DIA: The Dow Jones Industrial Average is comprised of 30 companies selected by a committee of Wall Street Journal editors.The only selection "rule" is companies must be substantial enterprises that represent a significant portion of the economic activity in the U.S. The Dow Jones Industrial Average is the 2nd oldest index dating back to 1896. The Dow Jones is a price-weighted index, meaning its value is derived from the price per share for each stock divided by a common divisor.
# 

# %% [markdown]
# # Load Data into Dataframes and modify

# %%
from formulas import *
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Read data in to dataframes

import yfinance as yf

spy = yf.Ticker("SPY")

# get stock info
spy.info

# get historical market data as df
hist = spy.history(period="max")

# Save df as CSV
hist.to_csv('../SPY.CSV')


spy = pd.read_csv('SPY.csv')
# dia = pd.read_csv('DIA.csv')
# qqq = pd.read_csv('QQQ.csv')


# Change date column to datetime type
spy['Date'] = pd.to_datetime(spy['Date'], utc=True).dt.strftime('%Y-%m-%d')
spy['Date'] = pd.to_datetime(spy['Date'])

# dia['Date'] = pd.to_datetime(dia['Date'])
# qqq['Date'] = pd.to_datetime(qqq['Date'])

# View amount of daily data
print(f'There are {spy.shape[0]} rows in SPY')
print('*'*100)
print(f'''The date range of SPY is {spy.index.min()} to {spy.index.max()} 
       ''')

# %%
# Plot history of adjusted close price of SPY
sns.lineplot(x = spy.Date, y=spy['Close'])
plt.title('SPY Daily Adjust Close Price')
plt.savefig('SPY Daily Adjust Close Price')

# %%
# Validate there is no duplicate dates
print(spy.index.is_unique)

# %% [markdown]
# ### Exclude 2020 Data
# For this excersise, and probably this entire project, we will only be using data up until Dec 31, 2019. This will give us a good complete yearly picture since we are not yet done with 2020.

# %%
# # Edit spy dataframe taking out all 2020 points
# spy = spy.loc[spy.Date.dt.year < 2020]

# # Edit dia dataframe taking out all 2020 points
# dia = dia.loc[dia.Date.dt.year < 2020]

# # Edit dia dataframe taking out all 2020 points
# qqq = qqq.loc[qqq.Date.dt.year < 2020]

# %% [markdown]
# ### Calculate Returns and View Distributions

# %% [markdown]
# It is better to make predictions of returns versus stock price. This is because a 2% gain on a 100 dollar stock is not the same as a 2% gain on a 10 dollar stock. Stock returns are stationary in the fact that the magnitude is always in the same relative range. We will do a distribution plot further in the notebook.

# %%
# compute daily return
spy['day_return'] = calculate_returns(spy['Close']) 
# dia['day_return'] = calculate_returns(dia['Adj Close'])

# compute daily log return
spy['log_day_return'] = compute_log_returns(spy['Close'])
# dia['log_day_return'] = compute_log_returns(dia['Adj Close'])

# Drop first row as the newly calculated columns will be N/A due to no previous data
spy = spy.iloc[1:]
# dia = dia.iloc[1:]

# Check out dataframe
spy.tail()

# %% [markdown]
# # Visualizing our Data

# %%
# print out daily return of the close adjusted price
sns.histplot(spy['day_return'])
print(spy['day_return'].agg(['mean','median','min','max','var', 'std']))

# %% [markdown]
# **DIA distribution**
# - Normal-ish distribution. High Kurtosis we will measure. (normal distribution mean = 0 std = 1)
# - Largest percent gain is 13.5% in one day
# - Largest loss in a day is -9%
# - Standard dev (volatility) is slightly larger than S&P500 probably due to less diversification
# 

# %%
# show daily return
# sns.distplot(dia['day_return'])
# print(dia['day_return'].agg(['mean', 'median', 'min','max','var', 'std']))

# %% [markdown]
# ### Day of the Week Return Analysis
# Ever wondered what day of the week offers the best upside? Does it always feel like friday has the buy button engaged after lunch and everyone takes off? Lets check.

# %%
# Create weekday column and map corresponding number output to real name
dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
spy['weekday'] = spy['Date'].dt.dayofweek.map(dayOfWeek)
# dia['weekday'] = dia['Date'].dt.dayofweek.map(dayOfWeek)

spy.head()

# %%
# Create SPY Barplot Showing 1 Day Return %
for m in range(1,13):
    print(m)
    sns.barplot(x="weekday", y="day_return", data=spy[spy['Date'].dt.month == m])
    plt.show()


# %% [markdown]
# #### Conclusions
# The two ETFs appear to behave very differently when looking at the returns for which day of the week it is. The DIA seems to not really care what day it ism
# 
# - Spy: Very interesting finds here. The SPY's best day is overwhelmingly Tuesday. It is almost as if Tuesday does not go down. The error bars are completely positive. Monday is the most volatile

# %% [markdown]
# ### Seasonal Trend Decomposition Plots

# %% [markdown]
# #### Two types of data from Trend Decomposition Plots
# 1. Systematic: Components of the time series that can be modeled due to their consistency
# 2. Non-Systematic: Components of the time series that are impossible to estimate/model
# 
# All time series data have 3 **systematic components called level, trend, seasonality**, and one **non-systematic component called noise.**
# 
# There are two types of time series:
# 1. Additive
# 2. Multiplicative
# 
# **Additive**<br/>
# Useful when trend and seasonal data is relatively constant overtime. This means the graphs looks linear.<br/>
# - *Target(t) = Level + Trend + Seasonality + Noise*
# 
# **Multiplicative**<br/>
# Useful when the trend and seasonal variation increases/decreases in magnitude over time. This can be seen by observing a  curved trend line. Non-linear seasonality is observed by noting increased distance from peak to trough of seasonality graphs or increased frequency of peak to trough within a period.<br/>
# - *Target(t) = Level * Trend * Seasonality * Noise*
# 

# %%
stl_spy = spy
stl_spy = stl_spy.set_index('Date')
stl_spy = stl_spy.resample('ME').last()
stl_spy = stl_spy['Close']

# %%
spy['Close'].head()

# %%
seasonal_trend_decomp_plot(dataframe = spy,
                           target_series = 'Close',
                           freq = 'ME',
                           seasonal_smoother = 13,
                           period = 12)

# %% [markdown]
# **Conclusion**<br/>
# Each of these components are something you may need to think about and address during data preparation, model selection, and model tuning. You may address it explicitly in terms of modeling the trend and subtracting it from your data, or implicitly by providing enough history for an algorithm to model a trend if it may exist.
# 
# Real-world problems are messy and noisy. There may be additive and multiplicative components. There may be an increasing trend followed by a decreasing trend. There may be non-repeating cycles mixed in with the repeating seasonality components.
# 
# - The trend looks to be nonlinear
# - The seasonality looks to multiplicative as the frequency increases and the magnitude increases
# - You can see the error (residuals) really begin to increase at the start of 2017

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Autocorrelation

# %%
from pandas.plotting import autocorrelation_plot

# Plot autocorrelation for SPY
autocorrelation_plot(spy['Adj Close'][:30])

# %%
# Plot autocorrelation for DIA
autocorrelation_plot(dia['Adj Close'][:10])

# %% [markdown]
# **Conlcusion**
# 
# - Seems to be positive until about 1750 day lag
# - the earlier more recent the point the better the correlation
# - No need to do anywhere the amount of lags we did. Need to focus earlier on

# %% [markdown]
# ### Lets Test first 10 days

# %%
# Establish empty list for for-loop
spy_autocorr = []

# Loop through lag numbers to find the point with highest autocorrelation
for x in list(range(1,101)):
    spy_autocorr.append(spy['Adj Close'].autocorr(lag=x))

# Show the index of the point with the largest autocorrelation value
# 0 indexed so add 1 for correct number of lags
lags = spy_autocorr.index(max(spy_autocorr)) + 1 
print('The best performing lag is number {}'.format(lags))

# %% [markdown]
# **Conclusion**
# 
# No need to do this for the DIA as well. This turned out exactly like our previous graph showed. An almost linear relationship with decreasing autocorrelation until around lag 2000. This means that the previous day is the best day to use to predict the next days price. However, with the average % change from the day before being almost 0, its not a very good indicator.

# %% [markdown]
# ### Arima Model

# %%
from statsmodels.tsa.arima_model import ARIMA

# fit model
spy_model = ARIMA(spy['Adj Close'], order=(5,1,0))
spy_model_fit = spy_model.fit(disp=0)
print(spy_model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(spy_model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

# %%
cal

# %%
spy['sma_3'] = spy.Close.rolling(window=3).mean()

# %%
spy.head()

# %%



