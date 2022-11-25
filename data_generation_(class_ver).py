import pandas as pd
import backtrader as bt
import yfinance as yf
import backtrader.feeds as btfeeds
from datetime import datetime
import ssl

#ssl._create_default_https_context = ssl._create_unverified_context

# Not-working code below:
# data = btfeeds.YahooFinanceData(dataname="SPY", fromdate=datetime(2016, 6, 25), todate=datetime(2021, 6, 25))


class LoadData():
    def __init__(self, start_train='2010-1-1', end_train="2018-1-1", start_test="2018-1-2", end_test="2020-1-1",
                 start_oos="2020-1-2", end_oos='2022-1-1'):
        self.start_train = start_train
        self.end_train = end_train
        self.start_test = start_test
        self.end_test = end_test
        self.start_oos = start_oos
        self.end_oos = end_oos

    def load(self):
        # Read the stock tickers that make up S&P500
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

        # Get the data for these tickers from yahoo finance
        data = yf.download(tickers.Symbol.to_list(), self.start_train, self.end_oos)
        data_close = data['Adj Close']
        # drop those stock that have no data for train period (490 stocks), they would not be fitted
        valid_c = data_close[data_close.index <= datetime.strptime(self.end_train, "%Y-%m-%d")].dropna(axis=1, how='all').columns
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
            data_is[c] = data_temp[data_temp.index <= datetime.strptime(self.end_test, "%Y-%m-%d")]
            data_oos[c] = data_temp[data_temp.index >= datetime.strptime(self.start_oos, "%Y-%m-%d")]
            data_train[c] = data_temp[data_temp.index <= datetime.strptime(self.end_train, "%Y-%m-%d")]
            data_test[c] = data_is[c][data_is[c].index >= datetime.strptime(self.start_test, "%Y-%m-%d")]
        return valid_c, data_all, data_is, data_oos, data_train, data_test

    def load_to_bt(self):
        valid_c, data_all, data_is, data_oos, data_train, data_test = LoadData.load(self)
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
        return data_train_bt, data_test_bt, data_is_bt, data_oos_bt


data = LoadData()
train_bt, test_bt, is_bt, oos_bt = data.load_to_bt()
