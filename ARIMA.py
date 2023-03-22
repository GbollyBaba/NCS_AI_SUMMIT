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
    df1 = data_content.pivot(index='TRADE_DATE', columns='SYMBOL', values='CLOSE_PRICE')
    df1.fillna(0.00)
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
    print('p-value:', result[1])
    print('Critical Values:')
    print('*********************************************************************')

    for key, value in result[4].items():

        print('\t%s: %.3f' % (key, value))
        #plt.text(15, max_y, 'ADF Statistic: {:.8f}'.format(result[0]), fontsize=12, ha='center', va='top')
        #plt.text(15, max_y+0.25, 'p-value: {:.8f}'.format(result[1]), fontsize=12, ha='center', va='top')
        #plt.text(15, max_y+0.5, 'Critical Values: {:.2f}'.format(result[2]), fontsize=12, ha='center', va='top')


    plt.savefig(x+"_stationarity.png")

    #test for stationarity and do a differncing  for the nonstationary series
    # Differencing to make the data stationary
    #diff = df.diff().dropna()

    # Visualize the differenced data
    #diff.plot()
    #plt.show()

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Plot the ACF and PACF of the differenced data
    #plot_acf(df1)
    #plot_pacf(df1)
    #plt.show()