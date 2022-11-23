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
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) :
        X = np.array(X)
        y = np.array(y)
    # form cumulative return
    X = (X-X[0])/X[0]
    y = (y-y[0])/y[0]
    muX = np.mean(X);muy = np.mean(y)
    c0 = np.inner(X-muX,y-muy)
    c1 = np.inner(X-muX,X-muX)-np.inner(y-muy,y-muy)
    c2 = -c1
    b1 = (-c1+(c1**2-4*c0*c2)**0.5)/(2*c0)
    b0 = muy - b1 * muX
    residual = (y-b1*X-b0)/(1+b1**2)**0.5
    return residual,b0,b1


class Kalman_filter():
    def __init__(self,data1,data2):
        # use x to predict y
        self.x = data1
        self.y = data2
        residual,b0,b1= tls(data2,data1)
        self.kf = KalmanFilter(initial_state_mean=np.mean(data2), 
                  transition_matrices = 1,observation_matrices = b1,
                  transition_offsets = 0,observation_offsets = b0,
                  n_dim_obs=1,n_dim_state=1)
        
        
    def get_prediction(self):
        #measurements = self.x
        n_timesteps = len(self.x)
        n_dim_state = 1
        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        for t in range(n_timesteps - 1):
            if t == 0:
                filtered_state_means[t] = np.mean(self.y)
                filtered_state_covariances[t] = [1]
            filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
                self.kf.filter_update(
                    filtered_state_means[t],
                    filtered_state_covariances[t],
                    self.x[t + 1]
                )
            )

        # draw estimates
        plt.figure()
        lines_true = plt.plot(self.y, color='b')
        lines_filt = plt.plot(filtered_state_means, color='r')
        plt.legend((lines_true[0], lines_filt[0]), ('true', 'filtered'))
        plt.show()
        
        return filtered_state_means

