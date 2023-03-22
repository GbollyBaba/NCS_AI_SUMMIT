import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller





#Load the data into a Pandas DataFrame:
data_content = pd.read_csv('Daily_Equity_Data_updated_001.csv', usecols=['TRADE_DATE',"SYMBOL", 'CLOSE_PRICE'])
data_content['TRADE_DATE'] = pd.to_datetime(data_content['TRADE_DATE'])
print(data_content.dtypes)
data = pd.read_excel('Daily_Equity_Data_updated_002.xlsx')
for x in data.SYMBOL.unique():
    #data_content = data_content[data_content['TRADE_DATE'] > pd.Timestamp('2015-05-29')]
    df1 = data_content.pivot(index='TRADE_DATE', columns='SYMBOL', values='CLOSE_PRICE')

    # perform differencing
    #df1['CLOSE_PRICE_diff1'] = df1[x].diff()
    #df1['CLOSE_PRICE_diff2'] = df1[x].diff()

    # drop the first row, which contains NaN due to differencing
    df1 = df1.dropna()


    df2 = df1[x]
    plt.plot(df1)
    plt.title(x + ' data from ' + str(np.min(data_content['TRADE_DATE']))[:10] + ' to ' + str(np.max(data_content['TRADE_DATE']))[:10]  )
    ts = pd.Series(df1[x])
    max_y = np.max(df1[x])
    # perform ADF test
    result = adfuller(ts)
    print('*********************************************************************')
    print(" " + x + "   ADFUller test for Stationarity")
    # print the results
    print('ADF Statistic:', result[0])
    adfstatistic = result[0]
    print('p-value:', result[1])
    p=result[1]
    print('Critical Values:')
    V = 1
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    # test fro staionarity  adfstatistic < siginificance levels

        if adfstatistic < value:
            V = V * 1
        else:
            V = V * 0
    if   V==1 and p<0.05:
        print(" Strong Stationary")
    if   V==1 and p>=0.05:
        print(" Weak Stationary")
    if V == 0 :
        print("Non-Stationary")


#p is \less than 0.05 t
#If the p-value is less than 0.05 and the ADF statistic is less than the critical values,
#you can reject the null hypothesis and conclude that the time series is stationary.



    print('*********************************************************************')

