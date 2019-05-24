import matplotlib
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import scipy
import matplotlib.pyplot as plt
import io, base64, os, json, re
import datetime
from random import randint
import statsmodels.api as sm
from statsmodels.formula.api import ols
import datetime as dt
style.use('ggplot')


start = dt.datetime(1980,1,1)
end = dt.datetime(2019,4,26)


#S&P 500 Data
SP500 = web.DataReader('^GSPC', 'yahoo', start, end)
SP500.to_csv('SP500.csv')
SP500 = pd.read_csv('SP500.csv')

SP500['Date'] = pd.to_datetime(SP500['Date'])



#Treasury Yield Rates
rates_df = pd.read_csv('USTREASURY-YIELD.csv')
rates_df['Date'] = pd.to_datetime(rates_df['Date'])
print(rates_df.head())

#Check Date Ranges
print('Rates:', np.min(rates_df["Date"]), np.max(rates_df['Date']))
print('SP500:', np.min(SP500["Date"]), np.max(SP500['Date']))

#Cut off excess data
cut_off_date = '1950-01-03'
SP500 = SP500[SP500['Date'] >= cut_off_date]
rates_df = rates_df[rates_df['Date'] >= cut_off_date]

print('SP500:', len(SP500), 'rates_df:', len(rates_df))



#Plot SP500
fig = plt.figure(figsize=(16, 8))
plt.plot(SP500['Date'], SP500['Adj Close'])
plt.suptitle('SP500')
plt.grid()
plt.show()




#Plot 3 yr vs 5 yr
fig = plt.figure(figsize=(16, 8))
plt.plot(rates_df['Date'],rates_df['5 YR'])
plt.plot(rates_df['Date'],rates_df['3 YR'])
plt.suptitle('rates_df')
plt.grid()
plt.show()


#Joining datasets
together = pd.merge(SP500[['Date', 'Adj Close']],
                    rates_df[['Date', '3 YR', '5 YR']],
                    on= ['Date'], how = 'left')
together = together.fillna(method='ffill')
together = together.dropna(axis=0)

print(together.tail(20))

#PLot SP500 and 3 yr 5yr yields
fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(together['Date'],
         together['3 YR'], color = 'r')
plt.plot(together['Date'],
         together['5 YR'], color = 'b')
plt.legend()
plt.grid()
plt.axhline(0)
ax.tick_params('vals', colors = 'r')

ax2 = ax.twinx()
plt.plot(together['Date'],
         together['Adj Close'],
         'c', label = 'S&P 500')
plt.legend()
plt.title('Rates VS S&P 500')
ax2.tick_params('vals', colors='b')
plt.show()

#Diff. 5 yr - 3 yr vs. SP500
fig, ax = plt.subplots(figsize=(16, 8))

plt.plot(together['Date'], together['5 YR']- together['3 YR'],
         color = 'r', label= '5 Yr. minus 3 Yr. Rates')
plt.legend()
plt.grid()
plt.axhline(0)
ax.tick_params('vals', colors='r')

ax2 = ax.twinx()
plt.plot(together['Date'],
         together['Adj Close'],
         'c', label= 'S&P 500')
plt.legend()
plt.title('Rates VS S&P 500')
ax2.tick_params('vals', colors='b')
plt.show()

together['3YR_PCT'] = together['3 YR'].pct_change()
together['5YR_PCT'] = together['5 YR'].pct_change()
together['SP500_PCT'] = together['Adj Close'].pct_change()
print(together.head())

tmp = together.copy()
cut_off_date = '1990-01-01'
tmp = tmp[tmp['Date'] > cut_off_date]

tmp['diff'] = tmp['5 YR'] - tmp['3 YR']

# join both datasets together
fig, ax = plt.subplots(figsize=(16, 8))

plt.plot(tmp['Date'], tmp['diff'].rolling(window=5).mean().values,
         color='r',
         linewidth=1, label='       Rates PCT')
plt.legend()
plt.grid()
plt.axhline(0)
ax.tick_params('vals', colors='r')
ax2 = ax.twinx()
plt.plot(tmp['Date'],
         tmp['Adj Close'].rolling(window=5).mean().values
         , 'c--', label='S&P 500 PCT')
plt.legend()
plt.title('Rates Diff VS S&P 500')
ax2.tick_params('vals', colors='b')


# background bar color
tmp['diff_simple'] = tmp['diff'].rolling(window=5).mean().values
tmp['diff_simple']  = [-100 if val > 0 else 100 for val in tmp['diff_simple'].values]
ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
              tmp['diff_simple'].values[np.newaxis],
              cmap='Paired', alpha=0.3)
plt.show()








#Unemployment rate
UR_Data = pd.read_csv('Unemployment Rate.csv')


UR_Data["Date"] = UR_Data["Year"].map(str) + UR_Data["Period"]
UR_Data['Date'] = UR_Data['Date'].str.replace('M01','-01-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M02','-02-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M03','-03-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M04','-04-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M05','-05-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M06','-06-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M06','-06-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M07','-07-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M08','-08-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M09','-09-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M10','-10-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M11','-11-01')
UR_Data['Date'] = UR_Data['Date'].str.replace('M12','-12-01')

UR_Data = UR_Data[['Date', 'Value']]
UR_Data['Date'] = pd.to_datetime(UR_Data['Date'])
UR_Data.columns = ['Date', 'UR']
print(UR_Data.tail())

fig = plt.figure(figsize=(16, 8))
plt.plot(UR_Data['Date'], UR_Data['UR'])
plt.suptitle('Unemployment Rate')
plt.grid()
plt.show()

#Merge Unemployment Data with S&P 500

URandSP500 = pd.merge(SP500[['Date', 'Adj Close']],
                    UR_Data[['Date', 'UR']],
                    on= ['Date'], how = 'left')
URandSP500 = URandSP500.fillna(method='ffill')
URandSP500 = URandSP500.dropna(axis=0)

URandSP500.columns = ['Date', 'Adj_Close', 'UR']

print(URandSP500.tail())

#PLot SP500 and Unemployment
fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(URandSP500['Date'],
         URandSP500['UR'], color = 'r')
plt.legend()
plt.grid()
plt.axhline(0)
ax.tick_params('vals', colors = 'r')

ax2 = ax.twinx()
plt.plot(URandSP500['Date'],
         URandSP500['Adj_Close'],
         'c', label = 'S&P 500')
plt.legend()
plt.title('UR VS S&P 500')
ax2.tick_params('vals', colors='b')
plt.show()

Regress = ols('Adj_Close ~ UR', URandSP500).fit()

print(Regress.summary())


#Case-Shiller Data

CaseShiller_df = pd.read_csv('CSUSHPINSA.csv')
CaseShiller_df['DATE'] = pd.to_datetime(CaseShiller_df['DATE'])
CaseShiller_df.columns = ['Date', 'Case-Shiller']
print(CaseShiller_df.tail())

#Plot Case-Shiller Index
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(CaseShiller_df['Date'], CaseShiller_df['Case-Shiller'])
plt.grid()
plt.show()

#Percent Change in Case Shiller
CaseShiller_df['CS_Pct_Chg'] = CaseShiller_df['Case-Shiller'].pct_change().copy()

fig, ax = plt.subplots(figsize=(16,8))
plt.plot(CaseShiller_df['Date'], CaseShiller_df['CS_Pct_Chg'], c = 'green')
ax.tick_params('vals', colors='r')
plt.grid()
plt.axhline(0)
plt.show()

CSandSP500 = pd.merge(SP500,
                      CaseShiller_df[['Date', 'Case-Shiller', 'CS_Pct_Chg']],
                      on=['Date'], how='left')
CSandSP500 = CSandSP500.fillna(method='ffill')
CSandSP500 = CSandSP500.dropna(axis=0)

print(CSandSP500.tail())

#Plot S&P500 with Case-Shiller
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(CSandSP500['Date'], CSandSP500['Case-Shiller'], color = 'r')
plt.grid()
plt.axhline(0)
ax.tick_params('vals', colors='r')
ax2 = ax.twinx()
plt.plot(CSandSP500['Date'], CSandSP500['Adj Close'], 'c', label= 'S&P 500')
plt.legend()
plt.title('S&P 500 and Case-Shiller')
ax2.tick_params('vals', colors='b')
plt.show()



#Plot S&P500 Percent Change VS. Case-Shiller Percent Change
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(CSandSP500['Date'], CSandSP500['CS_Pct_Chg'].rolling(window=50).mean().values, color = 'r')
plt.grid()
plt.axhline(0)
ax.tick_params('vals', colors='r')
ax2 = ax.twinx()
plt.plot(CSandSP500['Date'],
         CSandSP500['Adj Close'].pct_change().rolling(window=50).mean().values,
         'c--', label='S&P 500')
plt.legend()
plt.title('CPI VS S&P 500')
ax2.tick_params('vals', colors='b')
plt.show()





#Case-Shiller S&P 500 Regressions
CSandSP500 = CSandSP500.rename(columns = {'Adj Close': 'Adj_Close'})
CSandSP500 = CSandSP500.rename(columns = {'Case-Shiller': 'Case_Shiller'})

Regress2 = ols('Adj_Close ~ Case_Shiller', CSandSP500).fit()
print(Regress2.summary())


#VIX vs. S&P 500
VIX = web.DataReader('^VIX', 'yahoo', start, end)
VIX.to_csv('VIX.csv')
VIX = pd.read_csv('VIX.csv')
VIX['Date'] = pd.to_datetime(VIX['Date'])

print(VIX.tail())
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(VIX['Date'], VIX['Adj Close'], color = 'b')
plt.grid()
ax.tick_params('vals', colors = 'r')
ax2 = ax.twinx()
plt.plot(SP500['Date'], SP500['Adj Close'], label= 'S&P 500')
plt.legend()
plt.title('S&P 500 vs. VIX')
ax2.tick_params('vals', colors='b')
plt.show()


VIX = VIX.rename(columns = {'Adj Close' : 'VIX_Adj_Close'})
SP500 = SP500.rename(columns = {'Adj Close' : 'SP_Adj_Close'})

VIXandSP500 = pd.merge(VIX[['Date', 'VIX_Adj_Close']],
                       SP500[['Date', 'SP_Adj_Close']],
                       on = ['Date'], how = 'left')
VIXandSP500 = VIXandSP500.fillna(method='ffill')
VIXandSP500 = VIXandSP500.dropna(axis=0)




Regress3 = ols('SP_Adj_Close ~ VIX_Adj_Close', VIXandSP500).fit()
print(Regress3.summary())








