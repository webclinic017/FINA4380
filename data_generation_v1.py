import pandas as pd
import backtrader as bt
import yfinance as yf
import backtrader.feeds as btfeeds
from datetime import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Not-working code below:
# data = btfeeds.YahooFinanceData(dataname="SPY", fromdate=datetime(2016, 6, 25), todate=datetime(2021, 6, 25))

# Read the stock tickers that make up S&P500
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

# Get the data for these tickers from yahoo finance
data = yf.download(tickers.Symbol.to_list(), '2010-1-1', '2022-1-1')
data_close = data['Adj Close']
# drop those stock that have no data for train period (490 stocks), they would not be fitted
valid_c = data_close[data_close.index <= datetime.strptime('2018-1-1', "%Y-%m-%d")].dropna(axis=1, how='all').columns
data_st = data.stack()
data_sw = data_st.swaplevel()
data_all = {}
data_is = {}
data_oos = {}
data_train = {}
data_test = {}
for c in valid_c:
    data_temp = data_sw.loc[c, ]
    data_all[c] = data_temp
    data_is[c] = data_temp[data_temp.index <= datetime.strptime("2020-1-1", "%Y-%m-%d")]
    data_oos[c] = data_temp[data_temp.index > datetime.strptime("2020-1-1", "%Y-%m-%d")]
    data_train[c] = data_temp[data_temp.index <= datetime.strptime("2018-1-1", "%Y-%m-%d")]
    data_test[c] = data_is[c][data_is[c].index >= datetime.strptime("2018-1-2", "%Y-%m-%d")]
    # code to get csv's
    """
    data_is[c].to_csv('data_is_' + c + '.csv')
    data_oos[c].to_csv('data_oos_' + c + '.csv')
    data_train[c].to_csv('data_train_' + c + '.csv')
    data_test[c].to_csv('data_test_' + c + '.csv')
    """

# code to import to bt
data_train_bt = {}
data_test_bt = {}
data_is_bt = {}
data_oos_bt = {}
for c in valid_c:
    # train test split:　80:20
    data_train_bt[c] = btfeeds.PandasData(dataname=data_train[c])
    data_test_bt[c] = btfeeds.PandasData(dataname=data_test[c])
    data_is_bt[c] = btfeeds.PandasData(dataname=data_is[c])
    data_oos_bt[c] = btfeeds.PandasData(dataname=data_oos[c])

# code to get bt_standard csv by writers, very slow
"""
for c in valid_c:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000)
    cerebro.adddata(data_train_bt[c])
    cerebro.addwriter(bt.WriterFile, csv=True, out='data_train_bt_' + c + '.csv')
    cerebro.run()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000)
    cerebro.adddata(data_test_bt[c])
    cerebro.addwriter(bt.WriterFile, csv=True, out='data_test_bt_' + c + '.csv')
    cerebro.run()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000)
    cerebro.adddata(data_is_bt[c])
    cerebro.addwriter(bt.WriterFile, csv=True, out='data_is_bt_' + c + '.csv')
    cerebro.run()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000)
    cerebro.adddata(data_oos_bt[c])
    cerebro.addwriter(bt.WriterFile, csv=True, out='data_oos_bt_' + c + '.csv')
    cerebro.run()
"""
