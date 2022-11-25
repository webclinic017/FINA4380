from pykalman import KalmanFilter
from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import numpy.linalg as la
import math
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller

def tls(X,y):
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
    b1 = (-c1+(c1**2-4*c0*c2)**0.5)/(2*c0)
    b0 = muy - b1 * muX
    residual = (y-b1*X-b0)/(1+b1**2)**0.5
    return residual,b0,b1


class KalmanFilterPairs():        
    def __init__(self,data1,data2):
        # use x as para, y as observable variable
        self.x = data1
        self.y = data2
        residual,b0,b1= tls(data2,data1)
        self.b1
        # use the result from tls as observation_matrices
        self.kf = KalmanFilter(initial_state_mean=[b0,b1], 
                  transition_matrices = [[1, 0], [0, 1]],observation_matrices = [1,self.x[0]],
                  transition_offsets = [0,0],observation_offsets = 0,
                  n_dim_obs=1,n_dim_state=2)
        
        n_timesteps = len(data1)
        n_dim_state = 2
        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        for t in range(n_timesteps - 1):
            if t == 0:
                filtered_state_means[t] = [b0,b1]
                filtered_state_covariances[t] = [0.0001,0.0001]
            filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
                self.kf.filter_update(
                    filtered_state_means[t],
                    filtered_state_covariances[t],
                    self.x[t + 1]
                )
            )
        self.previous_cov = filtered_state_covariances[-1]
        
    def iter(self,d1,d2):
        prediction, pred_covariances = (
            self.kf.filter_update(
                observation_matrix = [1,d1],transition_covariance = self.previous_cov,
                transition_offset = 0,observation_offset = 0,observation = d2,
                filtered_state_mean = self.b1,filtered_state_covariance = self.previous_cov
                )
            )
        self.previous_cov = pred_covariances
        pred_d2 = np.inner(prediction,[1,d1])
        pred_error = d2 - pred_d2
        return pred_d2 ,pred_error,pred_covariances
    
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
data = yf.download(tickers.Symbol.to_list(), '2011-01-01', '2012-01-01')["Close"].dropna(axis=1)
KalmanFilterPairs(data.iloc[:,18],data.iloc[:,34])

