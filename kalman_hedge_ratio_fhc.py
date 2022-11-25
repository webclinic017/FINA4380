from pykalman import KalmanFilter
from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import numpy.linalg as la
import math
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
import datetime




class KalmanFilterPairs():        
    def __init__(self):
        # use x as para, y as observable variable
        
        # data2 and data3 is for initializing b0 and b1
        # self.residual = self.b0b1()['residual']
        # self.b0 = self.b0b1['b0']
        # self.b1 = self.b0b1['b1']

        train_startdate = datetime.datetime(2010, 1, 1)
        train_enddate = datetime.datetime(2017, 12, 31)
        self.x = np.array(yf.download('are', train_startdate, train_enddate, auto_adjust=True)['Close'])
        self.y = np.array(yf.download('tgt', train_startdate, train_enddate, auto_adjust=True)['Close'])
        self.residual, self.b0, self.b1 = self.tls(self.x, self.y)

        
        # use the result from tls as observation_matrices
        self.kf = KalmanFilter(initial_state_mean=[self.b0, self.b1], 
                  transition_matrices = [[1, 0], [0, 1]],observation_matrices = [1,self.x[0]],
                  transition_offsets = [0,0],observation_offsets = 0,
                  n_dim_obs=1,n_dim_state=2)
        
        n_timesteps = len(self.x)
        n_dim_state = 2
        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        for t in range(n_timesteps - 1):
            if t == 0:
                filtered_state_means[t] = [self.b0, self.b1]
                filtered_state_covariances[t] = [0.0001,0.0001]
            filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
                self.kf.filter_update(
                    filtered_state_means[t],
                    filtered_state_covariances[t],
                    observation_matrix = np.array([[1],[self.x[t]]]).T,
                    observation = self.x[t]
                )
            )
        # print(filtered_state_covariances)
        self.previous_cov = filtered_state_covariances[-1]
        self.previous_pred = filtered_state_means[-1]
    
    def tls(self, X, y):
        if isinstance(X, np.ndarray)==False:
            X = np.array(X)
            y = np.array(y)
        # form cumulative return
        X = np.log(X/X[0])
        y = np.log(y/y[0])
        muX = np.mean(X);muy = np.mean(y)
        c0 = np.inner(X-muX,y-muy)
        c1 = np.inner(X-muX,X-muX)-np.inner(y-muy,y-muy)
        c2 = -c1
        c3 = c1**2-4*c0*c2
        b1 = (-c1+math.sqrt(math.pow(c1,2)-4*c0*c2))/(2*c0)
        b0 = muy - b1 * muX
        residual = (y-b1*X-b0)/(1+b1**2)**0.5
        return residual, b0, b1
    # def b0b1(self):
    #         train_startdate = datetime.datetime(2010, 1, 1)
    #         train_enddate = datetime.datetime(2017, 12, 31)
    #         data2 = np.array(yf.download('are', train_startdate, train_enddate, auto_adjust=True))
    #         data3 = np.array(yf.download('tgt', train_startdate, train_enddate, auto_adjust=True))
    #         residual, b0, b1 = self.tls(data2, data3)
    #         print(residual)
    #         print(b0)
    #         return residual, b0, b1

    def iter(self,d1,d2):
        # print(self.previous_cov)
        prediction, pred_covariances = (
            self.kf.filter_update(
                observation_matrix = np.array([[1],[d1]]).T,
                observation = d2,
                filtered_state_mean = self.previous_pred,filtered_state_covariance = self.previous_cov
                )
            )
        self.previous_cov = pred_covariances
        self.previous_pred = prediction
        pred_d2 = np.inner(prediction,[1,d1])
        pred_error = d2 - pred_d2
        return pred_error,pred_covariances

# tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list()
# print(tickers[18])
# print(tickers[34])
# train_startdate = datetime.datetime(2010, 1, 1)
# train_enddate = datetime.datetime(2017, 12, 31)

# test_startdate = datetime.datetime(2018, 1, 1)
# test_enddate = datetime.datetime(2019, 12, 31)


# data0= yf.download("ARE", test_startdate, test_enddate)["Close"]
# data1= yf.download("TGT", test_startdate, test_enddate)["Close"]
# # data2= yf.download("ARE", train_startdate, train_enddate)["Close"]
# # data3= yf.download("TGT", train_startdate, train_enddate)["Close"]
# # print(data0)
# # print(data0)
# kalman_class = KalmanFilterPairs()
# pred_error, pred_covariance = kalman_class.iter(data0[0], data1[0])
# print(pred_error)
# print(pred_covariance)

# data = yf.download(tickers.Symbol.to_list(), '2011-01-01', '2012-01-01')["Close"].dropna(axis=1)
# kalman_class = KalmanFilterPairs(data.iloc[:,18],data.iloc[:,34])

# print(kalman_class.x)

# print(data0[0])