import numpy as np
import pickle
from sklearn import linear_model, metrics
import scipy as sp
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import cvxopt as opt
from cvxopt import blas, solvers


def load_object(file_name):
    """load the pickled object"""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def view_data(data_path):
    data = load_object(data_path)
    prices = data['prices']
    names = data['features']['names']
    features = data['features']['values']
    print(prices.shape)
    print(names)
    print(features.shape)
    return prices, features


# WEIGHT CHOOSING

def rolling_vol(prices, index):
    eq_price = pd.DataFrame(data = prices[:, index])
    std = eq_price.rolling(window = 10, min_periods = 1).std()
    std = np.concatenate(std.values, axis=0 )
    std = np.nan_to_num(std)
    return std


def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


def vol_pred(y):
    pred_vol = []

    print (y.shape)
    for i in range(len(y[0:])):
        vol = rolling_vol(y, i)
        predicted = StartARIMAForecasting(vol, 1,1,0)
        pred_vol.append(predicted[0])
    return pred_vol


def optimal_portfolio(returns, expected_ret, pred_vol):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    a = np.cov(returns)

    np.fill_diagonal(a, 0)

    mod_cov = a + np.diag(pred_vol)
    # Convert to cvxopt matrices
    S = opt.matrix(mod_cov)
    pbar = opt.matrix(expected_ret)


    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


def get_weights(expected_ret, y):
    volpred = vol_pred(y)
    weights, returns, risks = optimal_portfolio(y, expected_ret, volpred)
    return weights


# OLS MODELS

def train_OLS_models(x, y):
    models = []
    for i in range(680):
        model = sk.linear_model.LinearRegression(normalize=True)
        model.fit(x[i], y[i])
        models.append(model)
    return models

def test_OLS_models(models, x, y):
    for i in range(680):
        model = models[i]
        pred = model.predict(x[i])
        print("mae:", sklearn.metrics.mean_absolute_error(pred, y[i])/len(y[i])*100)


# BAYESIAN MODELS

def train_Bayesian_models(x, y):
    # prior param
    mu_0 = np.zeros(x.shape[-1])
    lambda_0 = np.identity(x.shape[-1])
    eta_0 = x.shape[-1]+1
    s_0 = np.identity(x.shape[-1])
    nu_0 = 2
    sigma_sq_0 = 1
    
    # sampler
    beta_0_samples, sigma_0_samples = [], []
    beta_j_samples, sigma_sq_samples = [], []

    # initial values
    beta_0_n = np.transpose(np.random.multivariate_normal(mu_0, lambda_0))
    sigma_0_n = np.linalg.inv(sp.stats.wishart.rvs(eta_0, s_0))
    beta_j_n = [np.transpose(np.random.multivariate_normal(beta_0_n, sigma_0_n)) for j in range(680)]
    sigma_sq_n = 1/np.random.gamma(nu_0/2, nu_0*sigma_sq_0/2)

    beta_0_samples.append(beta_0_n)
    sigma_0_samples.append(sigma_0_n)
    beta_j_samples.append(beta_j_n)
    sigma_sq_samples.append(sigma_sq_n)

    for i in range(2, 1500):
        # beta_j
        beta_j_n = []
        for j in range(680):
            vbeta_j = np.linalg.inv(np.linalg.inv(sigma_0_n) + np.matmul(np.transpose(x[j]), x[j])/sigma_sq_n)
            ebeta_j = np.matmul(vbeta_j, np.matmul(np.linalg.inv(sigma_0_n), beta_0_n) + np.matmul(np.transpose(x[j]), y[j])/sigma_sq_n)
            beta_j_n.append(np.transpose(np.random.multivariate_normal(ebeta_j, vbeta_j)))
        beta_j_samples.append(beta_j_n)
        
        # sigma_sq
        nu_n = nu_0 + 680*x.shape[1]
        ss = nu_0*sigma_sq_0
        for j in range(680):
            ss += sum((y[j] - np.matmul(x[j], beta_j_n[j]))**2)
        sigma_sq_n = 1/np.random.gamma(nu_n/2, ss/2)
        sigma_sq_samples.append(sigma_sq_n)
        
        # beta_0
        vbeta_0 = np.linalg.inv(np.linalg.inv(lambda_0) + 680*np.linalg.inv(sigma_0_n))
        ebeta_0 = np.matmul(vbeta_0, np.matmul(np.linalg.inv(lambda_0), mu_0) + np.matmul(np.linalg.inv(sigma_0_n), np.sum(beta_j_n, axis=0)))
        beta_0_n = np.transpose(np.random.multivariate_normal(ebeta_0, vbeta_0))
        beta_0_samples.append(beta_0_n)
        
        # sigma_0
        esigma_0 = eta_0 + 680
        s_beta_0 = np.zeros((10, 10))
        for j in range(680):
            s_beta_0 = np.add(s_beta_0, np.matmul(beta_j_n[j] - beta_0_n, np.transpose(beta_j_n[j] - beta_0_n)))
        ss = s_0 + s_beta_0
        sigma_0_n = np.linalg.inv(sp.stats.wishart.rvs(esigma_0, np.linalg.inv(ss)))
        sigma_0_samples.append(sigma_0_n)
    
    models = []
    for n in range(680):
        model = beta_j_samples[1000][n]
        for j in range(1001, len(beta_j_samples)):
            model = np.add(model, beta_j_samples[j][n])
        model = model/500
        models.append(model)
    
    return models


def test_Bayesian_models(models, x, y):
    for i in range(680):
        model = models[i]
        pred = [np.dot(model, x[i][j]) for j in range(len(x[i]))]
        print("mae:", sklearn.metrics.mean_absolute_error(pred, y[i])/len(y[i])*100)


class Strategy():
    def __init__(self):
        data = load_object("data/C3_train.pkl")
        data_prices = data['prices']
        data_features = data['features']['values']

        self.factors = np.array([[[data_features[j][k][i] for i in range(10)] for j in range(756)] for k in range(680)])
        self.prices = np.array([data[prices[i][j] for i in range(757)] for j in range(680)])
        self.returns = np.array([[(data_prices[i][j]-data_prices[i-1][j])/data_prices[i-1][j] for i in range(1, 757)] for j in range(680)])

        with open('bayesian_weights.pkl', 'rb') as f:
            self.models = pickle.load(f)

    def handle_update(self, inx, price, factors):
        """
        Args:
            inx: zero-based index in days
            price: [num_assets]
            factors: [num_assets, num_factors]
        Return:
            allocation: [num_assets]
        """

        # update data
        for i in range(680):
            self.factors[i].append(factors[i])
            self.prices[i].append(price[i])
            self.returns[i].append((self.prices[i][-1]-self.prices[i][-2])/self.prices[i][-2])

        # update model
        if inx % 21 == 0 and inx != 0:
            self.models = train_Bayesian_models(self.factors, self.returns)

        # predict returns
        expected_ret = []
        for i in range(680):
            expected_ret.append(np.dot(self.models[i], factors[i]))

        # update weights
        weights = get_weights(expected_ret, self.returns)
        return weights
