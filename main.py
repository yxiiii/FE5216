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

# ---------------Modify Log---------------
# 1. sample from 100000 -> 100000 [slightly improvement]
# 2. StandardScaler -> MaxminScaler [no much difference]
# 3. remove S0, K, log( S0 / K ) -> S0 / K:
#       GBM [LR from 0.89 -> 0.95, ANN from 0.95+ -> 0.9996]
#       Hestion [LR from 0.9 -> 0.97, ANN from 0.95+ -> 0.996]
# 4. Heston simulated path 500 -> 1000 [no much difference]
# 5. give dimension to accelerate data generating [10 times faster]


from MonteCarlo import MonteCarlo
import pandas as pd
import numpy as np
from tqdm import tqdm 
import random
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Learn():
    def __init__(self, model='GBM', Type='European', option='call'):
        self.model = model
        self.Type = Type
        self.option = option

    def simulate(self):

        model = self.model 
        Type = self.Type 
        option = self.option

        American = False if Type == 'European' else True
        optionType = 'c' if option == 'call' else 'p'

        if model == 'GBM':
            cols = ['moneyness', 'tau', 'r', 'vol', 'option price']
            data = []
            for i in tqdm(range(10000)):
                moneyness = random.uniform(0.5, 1.5)
                tau = random.uniform(0.3, 0.95)
                r = random.uniform(0.03, 0.08)
                vol = random.uniform(0.02, 0.9)
                # log_moneyness = np.log( moneyness ) 
                MC = MonteCarlo(model='GBM', S0=1, K=1/moneyness, T=tau, r=r, q=0, v=vol, method='MC')
                MC.generate_S(1000,10)
                option_price = MC.pricer(optionType=optionType, American=American)
                data.append([moneyness, tau, r, vol, option_price])
        
        elif model == 'Heston':
            cols = ['moneyness', 'tau', 'r', 'vol0', 'theta', 'kappa', 'gamma', 'rho', 'option price']
            data = []
            for i in tqdm(range(10000)):
                moneyness = random.uniform(0.6, 1.4)
                tau = random.uniform(0.1, 1.4)
                r = random.uniform(0, 0.1)
                vol0 = random.uniform(0.05, 0.5)
                theta = random.uniform(0, 0.5)
                kappa = random.uniform(0, 2)
                gamma = random.uniform(0, 0.5)
                rho = random.uniform(-0.95, 0)
                # moneyness = np.log( moneyness )                      
                MC = MonteCarlo(model='Heston', S0=1, K=1/moneyness, T=tau, r=r, q=0, v0=vol0, theta=theta, kappa=kappa, gamma=gamma, rho=rho)
                MC.generate_S(1000,10)
                option_price = MC.pricer(optionType=optionType, American=American)
                data.append([moneyness, tau, r, vol0, theta, kappa, gamma, rho, option_price])

        else: 
            raise(ValueError(f'You input a not-supported model type: {model}'))

        data = pd.DataFrame(data, columns=cols)
        data.to_csv(f'data_{model}_{Type}_{option}.csv', index=False)

    # Use Linear Regression as benchmark
    def LR(self):

        model = self.model 
        Type = self.Type 
        option = self.option

        data = pd.read_csv(f'data_{model}_{Type}_{option}.csv')

        if model == 'GBM':
            x = data[['moneyness', 'tau', 'r', 'vol']]
            y = data['option price']
        if model == 'Heston':
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
        print(f'Type: {Type}, option: {option}, Model: {model}, LR r2 = {score}')

    def ANN(self):
        
        model = self.model 
        Type = self.Type 
        option = self.option

        data = pd.read_csv(f'data_{model}_{Type}_{option}.csv')

        if model == 'GBM':
            x = data[['moneyness', 'tau', 'r', 'vol']]
            y = data['option price']
        if model == 'Heston':
            x = data[['moneyness', 'tau', 'r', 'vol0', 'theta', 'kappa', 'gamma', 'rho']]
            y = data['option price']

        # size of hidden layer
        x_num = x.shape[1]
        hls = (x_num * 2, x_num * 2, x_num * 2)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None, shuffle=True)

        StdScaler = StandardScaler()
        x_train = StdScaler.fit_transform(x_train)

        ANN = MLPRegressor(hidden_layer_sizes=hls, activation='relu', solver='adam', batch_size=1024, learning_rate_init=0.02, max_iter=500, tol=1e-10, shuffle=True, verbose=False, validation_fraction=0.1)
        ANN.fit(x_train, y_train)

        StdScaler = StandardScaler()
        x_test = StdScaler.fit_transform(x_test)

        score = ANN.score(x_test, y_test)
        print(f'Type: {Type}, option: {option}, Model: {model}, ANN r2 = {score}')

if __name__ == '__main__':

    model = input('GBM or Heston?(g/h)')
    Type = input('European or American?(e/a)')
    option = input('Call or Put?(c/p)')
    model = 'GBM' if model == 'g' else 'Heston'
    Type = 'European' if Type == 'e' else 'American'
    option = 'call' if option == 'c' else 'put'

    learn = Learn(model=model, Type=Type, option=option)
    # learn.simulate()
    learn.LR()
    learn.ANN()




