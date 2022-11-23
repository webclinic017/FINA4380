from pykalman import KalmanFilter
from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import numpy.linalg as la
import math
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
data = yf.download(tickers.Symbol.to_list(), '2011-01-01', '2012-01-01')["Close"].dropna(axis=1)

def tls(X,y):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) :
        X = np.array(X)
        y = np.array(y)
    muX = np.mean(X);muy = np.mean(y)
    c0 = np.inner(X-muX,y-muy)
    c1 = np.inner(X-muX,X-muX)-np.inner(y-muy,y-muy)
    c2 = -c1
    b1 = (-c1+(c1**2-4*c0*c2)**0.5)/(2*c0)
    b0 = muy - b1 * muX
    residual = (y-b1*X-b0)/(1+b1**2)**0.5
    return residual,b0,b1

test_result = np.zeros([data.shape[1],data.shape[1]])
for i in range(data.shape[1]):
    for j in range(i+1,data.shape[1]):
        residual,b0,b1= tls(data.iloc[:,i],data.iloc[:,j])
        if math.isnan(residual[0]): continue
        ADF =  adfuller(residual,autolag='t-stat')[0]
        test_result[i,j] = ADF
        
def find_min_index(matrix):
    pairs = []
    for i in range(10):
        print("the require ADF is {}".format(np.min(matrix)))
        tem_pair = np.where(matrix==np.min(matrix))
        tem_pair = [tem_pair[0][0],tem_pair[1][0]]
        pairs.append(tem_pair)
        matrix[tem_pair[0],:]=0;matrix[:,tem_pair[0]]=0
        matrix[tem_pair[1],:]=0;matrix[:,tem_pair[1]]=0
    return pairs

pairs = find_min_index(test_result)
# recheck for corr
for i in pairs:
    print(i)
    print(pearsonr(data.iloc[:,i[0]],data.iloc[:,i[1]])[0])
    
pd.DataFrame(pairs).to_csv("sample_pair.csv")