#%%
# a = MonteCarlo(model='GBM', S0=10, K=11, T=1, r=0.03, q=0, v=0.3)
# b = a.generate_S(5,20)
# c = a.pricer(optionType='c')

# a = MonteCarlo(model='Heston', S0=10, K=11, T=1, r=0.03, q=0, v0=0.2, theta=0.15, kappa=8, gamma=0.5, rho=0.2)
# b = a.generate_S(5,20)
# c = a.pricer(optionType='c')
import numpy as np
import pandas as pd

class MonteCarlo:
    def __init__(self, model, **kwargs):
        # basic params
        self.S0 = kwargs['S0']
        self.K = kwargs['K']
        self.T = kwargs['T']
        self.r = kwargs['r'] 
        self.q = kwargs['q'] 
        # extra params
        if model == 'GBM':
            self.modelType = 'GBM'
            # constant volatility
            self.v = kwargs['v']
        if model == 'Heston':
            self.modelType = 'Heston'
            # initial volatility
            self.v0 = kwargs['v0']
            # long-term average volatility
            self.theta = kwargs['theta']
            # recover speed
            self.kappa = kwargs['kappa']
            # volatility of volatility
            self.gamma = kwargs['gamma']
            # correlation between two brownian motions
            self.rho = kwargs['rho']
    
    def generate_S(self, n_paths, n_steps, antiVar=True):
        # list basic params needed
        self.n_paths = n_paths
        self.n_steps = n_steps
        dt = self.T / n_steps
        r = self.r - self.q
        S0 = self.S0
        
        if self.modelType == 'GBM':
            # list extra params needed
            v = self.v
            
            dw = np.random.normal(size=(n_paths, n_steps))
            if antiVar == True:
                n_paths *= 2
                self.n_paths = n_paths
                dw = np.concatenate((dw, -dw), axis=0)
            w = np.cumsum(dw, axis=1)
            # BS closed-form formula
            log_S = np.log(S0) \
                    + (r - v ** 2 / 2) * dt *  np.arange(1, n_steps + 1) \
                    + v * np.sqrt(dt) * w
            S = np.exp(log_S)
            S0 = (np.ones(n_paths) * S0).reshape(-1, 1)
            S = np.concatenate((S0, S), axis=1) 
            self.S = S 
        if self.modelType == 'Heston':
            # list extra params needed
            rho = self.rho
            v0 = self.v0
            theta = self.theta
            kappa = self.kappa
            gamma = self.gamma

            dw_v = np.random.normal(size=(n_paths, n_steps))
            dw_s = np.random.normal(size=(n_paths, n_steps))
            dw_s = rho * dw_v + np.sqrt(1 - rho ** 2) * dw_s 
            if antiVar == True:
                n_paths *= 2
                self.n_paths = n_paths
                dw_v = np.concatenate((dw_v, -dw_v), axis=0)
                dw_s = np.concatenate((dw_s, -dw_s), axis=0)
            
            # Euler–Maruyama
            v = np.zeros((n_paths, n_steps + 1))
            v[:, 0] = v0
            S = np.zeros((n_paths, n_steps + 1))
            S[:, 0] = S0
            for i in range(n_steps):
                v[:, i+1] = v[:, i] \
                            + kappa * ( theta - v[:, i]) * dt \
                            + gamma * np.sqrt(v[:, i]) * np.sqrt(dt) * dw_v[:, i]
                v[:, i+1] = np.absolute(v[:, i+1])
                S[:, i+1] = S[:, i] \
                            + r * S[:, i] * dt \
                            + np.sqrt(v[:, i]) * S[:, i] * np.sqrt(dt) * dw_s[:, i]
            self.S = S    
        return S 
    
    def LS(self, optionType='c', func_list=[lambda x: x ** 0, lambda x: x], buy_cost=0, sell_cost=0):
        dt = self.T / self.n_steps
        r = self.r 
        q = self.q
        df = np.exp(- r * dt)
        df2 = np.exp(- (r - q) * dt)
        S = self.S
        n_paths = self.n_paths
        n_steps = self.n_steps

        pass

 
    def pricer(self, optionType='c', American=False):
        S = self.S
        K = self.K
        T = self.T
        r = self.r 
        q = self.q


        if optionType == 'c':
            f_payoff = lambda x: np.maximum(x - K, 0)
        elif optionType == 'p':
            f_payoff = lambda x: np.maximum(K - x, 0) 
        else: 
            raise(ValueError('option type should be c or p.'))

        if not American:
            payoff = f_payoff(S[:, -1])
            dc_payoff = payoff * np.exp(-r * T)
        else: 
            if optionType == 'c' and q == 0:
                payoff = f_payoff(S[: -1])
                dc_payoff = payoff * np.exp(-r * T)
            else:
                pass

        option_price = np.average(dc_payoff)
        self.option_price = option_price
        return option_price





        
             
                    



                



#%%

#%% 
def a(mode, *args, **kwargs):
    if mode == 'GBM':
        print(args[0],args[1],kwargs['A'])
    if mode == 'Heston':
        print(args[0],args[1],kwargs['A'])
    
# %%
import numpy as np 
np.random.normal(size=(2,3))
# %%
