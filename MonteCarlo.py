#%%
import numpy as np

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
            # use closed formula or MonteCarlo 
            self.method = kwargs['method'] 
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
        
        self.n_paths = n_paths
        self.n_steps = n_steps
        dt = self.T / n_steps
        r = self.r - self.q
        S0 = self.S0
        
        if self.modelType == 'GBM':
            # if we use closed-form formula, no need to generate S paths
            if self.method == 'formula':
                return
            # f we use Monte Carlo, generate S paths
            v = self.v
            dw = np.random.normal(size=(n_paths, n_steps))
            if antiVar == True:
                n_paths *= 2
                self.n_paths = n_paths
                dw = np.concatenate((dw, -dw), axis=0)
            w = np.cumsum(dw, axis=1)
            log_S = np.log(S0) \
                    + (r - v ** 2 / 2) * dt *  np.arange(1, n_steps + 1) \
                    + v * np.sqrt(dt) * w
            S = np.exp(log_S)
            S0 = (np.ones(n_paths) * S0).reshape(-1, 1)
            S = np.concatenate((S0, S), axis=1) 
            self.S = S 
        if self.modelType == 'Heston':
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
        return self.S
    
    def LS(self, payoff, k=2):   
        def LSLaguerre(k, x):
            if k == 0:
                return np.exp(-0.5 * x) * 1
            elif k == 1:
                return np.exp(-0.5 * x) * ( - x + 1 )
            elif k == 2:
                return np.exp(-0.5 * x) * (0.5 * ( x * x - 4 * x + 2 ))
            else:
                raise(ValueError('K at most 2'))

        dt = self.T / self.n_steps
        r = self.r 
        T = self.T

        S = self.S.T
        N1 = self.n_paths
        M = self.n_steps

        V = np.zeros((M+1, N1))
        V[-1, :] = np.exp( -r * T ) * payoff(S[-1, :])

        h = np.zeros((N1, k, M+1))
        hplus = np.zeros((k, N1, M+1))

        for i in range(k):
            h[:,i,:] = LSLaguerre(i, S).T

        for i in range(1,M):
            numerator = np.dot(h[:,:,i].T, h[:,:,i])
            hplus[:,:,i] = np.dot(np.mat(numerator).I.A, h[:,:,i].T)

        a = np.zeros((k, M+1))  # optimal weights 
        C = np.zeros((M+1, N1)) # continuation value

        for i in np.arange(M-1,-1,-1):    
            a[:, i] = np.dot(hplus[:,:,i], V[i+1, :].reshape(1,-1).T).reshape(-1,)
            for j in np.arange(0, N1):
                # estimation of continuation value
                C[i, j] = np.dot(a[:, i].reshape(-1,1).T, h[j,:,i].reshape(-1,1))
                if np.exp(-r * (i-1) * dt) * payoff(S[i,j]) > C[i,j]:
                    V[i,j] = np.exp(-r * i * dt) * payoff(S[i,j])
                else:
                    V[i,j] = V[i+1,j];

        return np.average(V[1,:])

    def pricer(self, optionType='c', American=False):

        S0 = self.S0
        K = self.K
        T = self.T
        r = self.r 
        q = self.q
        
        if self.modelType == 'GBM':
            if self.method == 'formula':
                from scipy.stats import norm 
                v = self.v
                N = norm.cdf
                d1 = ( np.log( S0 / K ) + T * ( r - q + v ** 2 / 2) ) / ( v * np.sqrt(T) )
                d2 = d1 - v * np.sqrt(T)
                option_price = S0 * N(d1) - np.exp(-r * T) * K * N(d2) 
                if option_price < 0:
                    print(N(d1), N(d2) )
                return option_price

        if optionType == 'c':
            f_payoff = lambda x: np.maximum(x - K, 0)
        elif optionType == 'p':
            f_payoff = lambda x: np.maximum(K - x, 0) 
        else: 
            raise(ValueError('option type should be c or p.'))

        S = self.S
        if ( not American ) or ( American and optionType == 'c' and q == 0 ):
            payoff = f_payoff(S[:, -1])
            dc_payoff = payoff * np.exp(-r * T)
            return np.average(dc_payoff)
        else:
            return self.LS(f_payoff, k=2)

