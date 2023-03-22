import numpy
import numpy as np
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


print("Relationship with NSE  BANKING STOCKS ")
print("***********************")
print("***********************")
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})
k = 1
dfAll=[]
dfAllTemp=[]
cols= []
data = pd.read_csv('/Users/akingboladeshada/Desktop/Daily_Equity_Data_updated_001.csv',usecols=["SYMBOL"])
print(data.dtypes)
for x in data.SYMBOL.unique():
    cols.append(x)

# for TSaionary

from statsmodels.tsa.stattools import adfuller
# perform ADF test on the time series

data_content = pd.read_csv('/Users/akingboladeshada/Desktop/Daily_Equity_Data_updated_001.csv', usecols=['TRADE_DATE',"SYMBOL", 'CLOSE_PRICE'])
data_content['TRADE_DATE'] = pd.to_datetime(data_content['TRADE_DATE'])
print(data_content.dtypes)

#data_content = data_content[data_content['TRADE_DATE'] > pd.Timestamp('2015-05-29')]
#data_content = data_content[data_content['TRADE_DATE'] < pd.Timestamp('205-05-29')]
df1 = data_content.pivot(index='TRADE_DATE', columns='SYMBOL', values='CLOSE_PRICE')

#fill na with 0.00
df1 =df1.fillna(0.00)

#Calculation and visualization of the covariance matrix
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(df1[cols].iloc[:,range(0,13)].values)
cov_mat =np.corrcoef(X_std.T)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
hm = sns.heatmap(cov_mat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 cmap='coolwarm',
                 yticklabels=cols,
                 xticklabels=cols)
plt.suptitle('Stock Price Corelation Among 14  Banks in Nigeria ', size = 18)
plt.title( str(np.min(data_content['TRADE_DATE'] )) + " to " + str(np.max(data_content['TRADE_DATE'] )), size = 14)

plt.tight_layout()
plt.show()
# Get the lower triangle of the covariance matrix using numpy.tril()
lower_tri = np.tril(cov_mat)



# print  covariance between threshold between 0.8 and less than 1.0

k= 1
print("*************************************************************************************")
print("*************************************************************************************")
print("*************************************************************************************")
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)-1):
        if abs(cov_mat[i, j]) > 0.5  and   abs(cov_mat[i, j]) < 1.00:
            print(str(k )  + ".    [" + cols[i] +" : " + cols[j] + "]    ,  COV="+ str(cov_mat[i, j]))

            print ("test for  long term relationship with Co Integartion")
            result = sm.tsa.stattools.coint(df1[cols[i]], df1[cols[j]])

            # print the test results
            print('Cointegration test results:')
            print('t-statistic:', result[0])
            print('p-value:', result[1])
            print('critial values:', result[2])
            p =result[1]
            V = 1
            tstatistic = result[0]
            for value in result[2]:
                print(' %.3f' % ( value))
                # test fro staionarity  adfstatistic < siginificance levels

                if tstatistic < value:
                    V = V + 1
                else:
                    V = V + 0
                    # i use + to handle OR
            if V >= 1 and p < 0.05:
                print(" Strong long-term relationship ")
            if V >= 1 and p >= 0.05:
                print(" Weak long-term relationship")
            if V < 1:
                print("NO long-term relationship")


            print("*************************************************************************************")








exit()