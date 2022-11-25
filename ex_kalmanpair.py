import backtrader as bt
import numpy as np
import datetime
import yfinance as yf
import pandas as pd
import backtrader.analyzers as btanalyzers
class KalmanPair(bt.Strategy):
    params = (("printlog", True), ("quantity", 1000), ("coef", 1))

    def log(self, txt, dt=None, doprint=False):
        """Logging function for strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        self.delta = 0.0001
        self.Vw = self.delta / (1 - self.delta) * np.eye(2)
        self.Ve = 0.001

        self.beta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = np.zeros((2, 2))

        self.position_type = None  # long or short
        self.quantity = self.params.quantity

        # keep reference to the close line in the series
        # self.dataclose0 = self.data0[0]
        # self.dataclose1 = self.data1[0]

    # def notify_order(self, order):
    #     if order.status in [order.Submitted, order.Accepted]:
    #         # Buy/Sell order submitted/accepted to/by broker - Nothing to do
    #         return

    #     # Check if an order has been completed
    #     # Attention: broker could reject order if not enough cash
    #     if order.status in [order.Completed]:
    #         if order.isbuy():
    #             self.log('BUY EXECUTED, %.2f' % order.executed.price)
    #         elif order.issell():
    #             self.log('SELL EXECUTED, %.2f' % order.executed.price)

    #         self.bar_executed = len(self)

    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:
    #         self.log('Order Canceled/Margin/Rejected')

    #     # Write down: no pending order
    #     self.order = None


    
    def next(self):

        x = np.asarray([self.data0[0], 1.0]).reshape((1, 2))
        y = self.data1[0]

        self.R = self.P + self.Vw  # state covariance prediction
        yhat = x.dot(self.beta)  # measurement prediction

        Q = x.dot(self.R).dot(x.T) + self.Ve  # measurement variance

        e = y - yhat  # measurement prediction error

        K = self.R.dot(x.T) / Q  # Kalman gain

        self.beta += K.flatten() * e  # State update
        self.P = self.R - K * x.dot(self.R)

        sqrt_Q = np.sqrt(Q)
        leverage = 0.06
        if self.position:
            if self.position_type == "long" and e > -self.params.coef*sqrt_Q:
                self.close(self.data0)
                self.close(self.data1)
                self.position_type = None
                self.log('LONG POSITION CLOSE, stock1 close price %.2f' % self.data0[0])
                self.log('LONG POSITION CLOSE, stock2 close price %.2f' % self.data1[0])
            if self.position_type == "short" and e < self.params.coef*sqrt_Q:
                self.close(self.data0)
                self.close(self.data1)
                self.position_type = None
                self.log('SHORT POSITION CLOSE, stock1 close price %.2f' % self.data0[0])
                self.log('SHORT POSITION CLOSE, stock2 close price %.2f' % self.data1[0])

        else:
            if e < -self.params.coef*sqrt_Q:
                self.sell(data=self.data0, size=leverage*(self.quantity * self.beta[0]))
                self.buy(data=self.data1, size=leverage*self.quantity)
                self.position_type = "long"
                self.log('LONG CREATE, sell stock1, close price %.2f' % self.data0[0])
                self.log('LONG CREATE, buy stock2 close price %.2f' % self.data1[0])
            if e > self.params.coef*sqrt_Q:
                self.buy(data=self.data0, size=leverage*(self.quantity * self.beta[0]))
                self.sell(data=self.data1, size=leverage*self.quantity)
                self.log('SHORT CREATE, buy stock1 close price %.2f' % self.data[0])
                self.log('SHORT CREATE, sell stock2 close price %.2f' % self.data[0])
                self.position_type = "short"

        # self.log(f"beta: {self.beta[0]}, alpha: {self.beta[1]}")


def run():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(KalmanPair)
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')
    startdate = datetime.datetime(2016, 1, 1)
    enddate = datetime.datetime(2020, 1, 1)
    symbol = pd.read_csv("futures.csv",index_col=False)
    symbol = list(symbol.iloc[:,0])
    pair =[1, 20]
    ewa = bt.feeds.PandasData(dataname=yf.download(symbol[pair[0]],startdate, enddate, auto_adjust=True))
    ewc = bt.feeds.PandasData(dataname=yf.download(symbol[pair[1]], startdate, enddate, auto_adjust=True))
    
    cerebro.adddata(ewa)
    cerebro.adddata(ewc)
    # cerebro.broker.setcommission(commission=0.0001)
    cerebro.broker.setcash(100_000.0)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    thestrats = cerebro.run()[0]
    print("the pair is {}, {}".format(symbol[pair[0]],symbol[pair[1]]))
    print('Sharpe Ratio:', thestrats.analyzers.mysharpe.get_analysis())
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()


if __name__ == "__main__":
    run()