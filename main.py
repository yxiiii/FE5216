'''
class MonteCarlo:

    if model = 'GBM': 
        params: 
            S0,     asset price at initial time
            K,      strike price
            T,      time to maturity
            r,      risk free rate
            q,      dividend rate
            v,      constant volatility
            method, use closed-form formula or Monte Carlo
    if model = 'Heston:
        params:
            S0,     asset price at initial time
            K,      strike price
            T,      time to maturity
            r,      risk free rate
            q,      dividend rate
            v0,     volatility at initial time
            theta,  long-term volatility
            kappa,  reversion speed
            gamma,  volatility of volatility
            rho,    correlation between two brownian motions

'''
# ---------------example---------------
# a = MonteCarlo(model='GBM', S0=10, K=11, T=1, r=0.03, q=0, v=0.3, method='formula')
# b = a.generate_S(5,20)
# c = a.pricer(optionType='c')

# a = MonteCarlo(model='Heston', S0=10, K=11, T=1, r=0.03, q=0, v0=0.2, theta=0.15, kappa=8, gamma=0.5, rho=0.2)
# b = a.generate_S(5,20)
# c = a.pricer(optionType='c')


#%%
from tabnanny import verbose
from MonteCarlo import MonteCarlo
import pandas as pd
import numpy as np
from tqdm import tqdm 
import random
import logging as logger
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
 
# GBM
# cols = ['moneyness', 'tau', 'r', 'vol', 'option price']
# data = pd.DataFrame(columns=cols)
# for i in tqdm(range(100000)):
#     S0 = random.uniform(10,20)
#     K = random.uniform(10,20)
#     tau = random.uniform(0.3, 0.95)
#     r = random.uniform(0.03, 0.08)
#     vol = random.uniform(0.02, 0.9)
#     moneyness = np.log( S0 / K ) 
#     MC = MonteCarlo(model='GBM', S0=S0, K=K, T=tau, r=r, q=0, v=vol, method='formula')
#     MC.generate_S(1000,10)
#     option_price = MC.pricer(optionType='c')
#     data = pd.concat([data, pd.DataFrame([[moneyness, tau, r, vol, option_price]], columns=cols)], axis=0)

# data.to_csv('data_GBM.csv', index=False)

# Heston
# cols = ['moneyness', 'tau', 'r', 'vol0', 'theta', 'kappa', 'gamma', 'rho', 'option price']
# data = pd.DataFrame(columns=cols)
# for i in tqdm(range(100000)):
#     S0 = random.uniform(10,20)
#     K = random.uniform(10,20)
#     tau = random.uniform(0.1, 1.4)
#     r = random.uniform(0, 0.1)
#     vol0 = random.uniform(0.02, 0.9)
#     theta = random.uniform(0, 0.5)
#     kappa = random.uniform(0, 2)
#     gamma = random.uniform(0, 0.5)
#     rho = random.uniform(-0.95, 0)
#     moneyness = np.log( S0 / K )                      
#     MC = MonteCarlo(model='Heston', S0=S0, K=K, T=tau, r=r, q=0, v0=vol0, theta=theta, kappa=kappa, gamma=gamma, rho=rho)
#     MC.generate_S(500,10)
#     option_price = MC.pricer(optionType='c')
#     data = pd.concat([data, pd.DataFrame([[moneyness, tau, r, vol0, theta, kappa, gamma, rho, option_price]], columns=cols)], axis=0)

# data.to_csv('data_Heston.csv', index=False)

# Use Linear Regression as benchmark
def LR(model='GBM'):
    if model == 'GBM':
        data = pd.read_csv('data_GBM.csv')
        x = data[['moneyness', 'tau', 'r', 'vol']]
        y = data['option price']
    if model == 'Heston':
        data = pd.read_csv('data_Heston.csv')
        x = data[['moneyness', 'tau', 'r', 'vol0', 'theta', 'kappa', 'gamma', 'rho']]
        y = data['option price']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None, shuffle=True)
    
    StdScaler = StandardScaler()
    x_train = StdScaler.fit_transform(x_train)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    StdScaler = StandardScaler()
    x_test = StdScaler.fit_transform(x_test)

    score = lr.score(x_test, y_test)
    print(f'Model: {model}, LR r2 = {score}')

def ANN(model='GBM'):
    if model == 'GBM':
        data = pd.read_csv('data_GBM.csv')
        x = data[['moneyness', 'tau', 'r', 'vol']]
        y = data['option price']
    if model == 'Heston':
        data = pd.read_csv('data_Heston.csv')
        x = data[['moneyness', 'tau', 'r', 'vol0', 'theta', 'kappa', 'gamma', 'rho']]
        y = data['option price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None, shuffle=True)

    StdScaler = StandardScaler()
    x_train = StdScaler.fit_transform(x_train)

    ANN = MLPRegressor(hidden_layer_sizes=(8), activation='logistic', solver='sgd', batch_size=1024, learning_rate='constant', learning_rate_init=0.02, max_iter=100, tol=1e-10, shuffle=True, verbose=False, validation_fraction=0.1)
    ANN.fit(x_train, y_train)

    StdScaler = StandardScaler()
    x_test = StdScaler.fit_transform(x_test)

    score = ANN.score(x_test, y_test)
    print(f'Model: {model}, ANN r2 = {score}')

LR('GBM')
ANN('GBM')
LR('Heston')
ANN('Heston')

# %%
