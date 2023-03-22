import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")

df_full = pd.read_csv('/Users/akingboladeshada/Desktop/NSE/NEWDATA.csv')

data = pd.read_excel('/Users/akingboladeshada/Desktop/NSE/Daily_Equity_Data_updated_002.xlsx')
for x in data.SYMBOL.unique():
    stock_data = pd.read_excel('/Users/akingboladeshada/Desktop/NSE/Daily_Equity_Data_updated_002.xlsx')
    data2021 = pd.read_excel('/Users/akingboladeshada/Desktop/NSE/mankind.xlsx')
    data2021 =data2021[(data2021["COMPANY"] == x)]
    stock_data = stock_data[(stock_data["SYMBOL"] == x)]
    df1 = pd.DataFrame(stock_data, columns=['TRADE_DATE', 'CLOSE_PRICE'])
    df_111 = pd.DataFrame(stock_data, columns=['TRADE_DATE', 'CLOSE_PRICE',
                                         'DIV', 'DIV_YIELD', 'EPS', 'PE_RATIO','PAYMENT_DATE'])

    #stock_data = yf.download('AAPL', start='2016-01-01', end='2021-10-01')
    print(stock_data.head())

    df = df_full[(df_full["SYMBOL"] == x)]
    df.TRADE_DATE = pd.to_datetime(df.TRADE_DATE)
    df = df.sort_values(by=['TRADE_DATE'])
    df = df.set_index("TRADE_DATE")
    df.drop(columns=['SYMBOL'],inplace=True)
    sizeINPUT = 30
    train, test = df[:-sizeINPUT], df[-sizeINPUT:]

    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    n_input = sizeINPUT
    n_features = 1
    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

    #model = Sequential()
    #model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
    #model.add(Dropout(0.15))
    #model.add(Dense(1))
    #model.compile(optimizer='adam', loss='mse')

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(n_input, n_features)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.add(Dropout(0.15))
    model.compile(optimizer='adam', loss='mse')
    model.summary()



    model.fit_generator(generator, epochs=2)

    pred_list = []

    batch = train[-n_input:].reshape((1, n_input, n_features))

    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)

    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=df[-n_input:].index, columns=['Prediction'])

    df_test = pd.concat([df, df_predict], axis=1)

    plt.figure(figsize=(20, 5))
    plt.title(x)
    plt.plot(df_test.index, df_test['CLOSE_PRICE'])
    plt.plot(df_test.index, df_test['Prediction'], color='r')
    plt.legend(loc='best', fontsize='xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.show()

    pred_actual_rmse = rmse(df_test.iloc[-n_input:, [0]], df_test.iloc[-n_input:, [1]])
    print("rmse: ", pred_actual_rmse)

    train = df

    scaler.fit(train)
    train = scaler.transform(train)

    n_input = 30
    n_features = 1
    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

    model.fit_generator(generator, epochs=1)

    pred_list = []
#https://www.youtube.com/watch?v=IebXmSJtGqo
    batch = train[-n_input:].reshape((1, n_input, n_features))

    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)

    from pandas.tseries.offsets import DateOffset

    add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0, n_input+1)]
    future_dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)

    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=future_dates[-n_input:].index, columns=['Prediction'])

    df_proj = pd.concat([df, df_predict], axis=1)

    plt.figure(figsize=(20, 5))
    plt.title(x)
    plt.plot(df_proj.index, df_proj['CLOSE_PRICE'])
    plt.plot(df_proj.index, df_proj['Prediction'], color='r')
    plt.legend(loc='best', fontsize='xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.show()

    df_proj.to_csv()
    df_proj.to_csv('/Users/akingboladeshada/Desktop/NSE/LSTM_' + x + '.csv')

    #