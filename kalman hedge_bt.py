import backtrader as bt
import numpy as np
import datetime
import yfinance as yf
import math
from kalman_hedge_ratio import KalmanFilterPairs

class KalmanPair(bt.Strategy):
    params = (("printlog", True), ("quantity", 1000), ("coef", 0.1))

    def log(self, txt, dt=None, doprint=False):
        """Logging function for strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        self.kf_class = KalmanFilterPairs()
        # self.delta = 0.0001
        # self.Vw = self.delta / (1 - self.delta) * np.eye(2)
        # self.Ve = 0.001

        # self.beta = np.zeros(2)
        # self.P = np.zeros((2, 2))
        # self.R = np.zeros((2, 2))

        self.position_type = None  # long or short
        self.quantity = self.params.quantity



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

        e, cov = self.kf_class.iter(self.data0[0], self.data1[0])
        sqrt_Q = math.sqrt(cov[1][1])
        self.log('ERROR:'+str(e))
        self.log('SQRT:'+str(sqrt_Q))
        

        # x = np.asarray([self.data0[0], 1.0]).reshape((1, 2))
        # y = self.data1[0]

        # self.R = self.P + self.Vw  # state covariance prediction
        # yhat = x.dot(self.beta)  # measurement prediction

        # Q = x.dot(self.R).dot(x.T) + self.Ve  # measurement variance

        # e = y - yhat  # measurement prediction error

        # K = self.R.dot(x.T) / Q  # Kalman gain

        # self.beta += K.flatten() * e  # State update
        # self.P = self.R - K * x.dot(self.R)

        # sqrt_Q = np.sqrt(Q)

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
                self.sell(data=self.data0, size=(self.quantity * self.beta[0]))
                self.buy(data=self.data1, size=self.quantity)
                self.position_type = "long"
                self.log('LONG CREATE, sell stock1, close price %.2f' % self.data0[0])
                self.log('LONG CREATE, buy stock2 close price %.2f' % self.data1[0])
            if e > self.params.coef*sqrt_Q:
                self.buy(data=self.data0, size=(self.quantity * self.beta[0]))
                self.sell(data=self.data1, size=self.quantity)
                self.log('SHORT CREATE, buy stock1 close price %.2f' % self.data[0])
                self.log('SHORT CREATE, sell stock2 close price %.2f' % self.data[0])
                self.position_type = "short"

        # self.log(f"beta: {self.beta[0]}, alpha: {self.beta[1]}")


def run():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(KalmanPair)

    test_startdate = datetime.datetime(2018, 1, 1)
    test_enddate = datetime.datetime(2019, 12, 31)

    data0 = bt.feeds.PandasData(dataname=yf.download('ewa', test_startdate, test_enddate, auto_adjust=True))
    data1 = bt.feeds.PandasData(dataname=yf.download('ewc', test_startdate, test_enddate, auto_adjust=True))


    data_list = [data0, data1]
    for i in data_list:
        cerebro.adddata(i)

    # cerebro.broker.setcommission(commission=0.0001)
    cerebro.broker.setcash(100_000.0)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()


if __name__ == "__main__":
    run()