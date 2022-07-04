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
        T = self.T
        dt = T / n_steps
        r = self.r
        q = self.q
        S0 = self.S0
        K = self.K
        
        if self.modelType == 'GBM':
            v = self.v
            if antiVar == True:
                dw = np.random.normal( size=(int(n_paths / 2), n_steps) )
                dw = np.concatenate((dw, -dw), axis=0)
            else:
                dw = np.random.normal( size=(n_paths, n_steps) )
            w = np.cumsum(dw, axis=1)
            log_S = np.log(S0) \
                    + (r - q - v ** 2 / 2) * dt *  np.arange(1, n_steps + 1) \
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

            if antiVar == True:
                dw_v = np.random.normal(size=(int(n_paths / 2), n_steps))
                dw_s = np.random.normal(size=(int(n_paths / 2), n_steps))
                dw_s = rho * dw_v + np.sqrt(1 - rho ** 2) * dw_s 
                dw_v = np.concatenate((dw_v, -dw_v), axis=0)
                dw_s = np.concatenate((dw_s, -dw_s), axis=0)
            else:
                dw_v = np.random.normal(size=(n_paths, n_steps))
                dw_s = np.random.normal(size=(n_paths, n_steps))
                dw_s = rho * dw_v + np.sqrt(1 - rho ** 2) * dw_s 
            
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
            self.v = v
            self.S = S    
    def get_path(self, type='v'):
        if type == 'v': 
            return self.v
        if type == 'S':
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

    def pricer(self, n_paths, n_steps, optionType='c', European=True, antiVar=True):
        '''
        if use GBM: 
            if price European option or American option but dividend rate is 0:
                generate S[T] (antithetic variates if necessary)
                return  discounted np.average(payoff(S[T], K))
        
        Other conditions we need to generate price paths first:
            1. use Heston
            2. use GBM but to price American option with dividend rate not equal to 0
        '''

        S0 = self.S0
        K = self.K
        T = self.T
        r = self.r 
        q = self.q
        
        if n_paths % 2:
            n_paths = n_paths +1

        if self.modelType == 'GBM':
            if European or ((not European) and (q == 0)):
                if antiVar == True:
                    dw = np.random.normal( size=(int(n_paths / 2), n_steps) )
                    dw = np.concatenate((dw, -dw), axis=0)
                else:
                    dw = np.random.normal( size=(n_paths, n_steps) )
                v = self.v
                logStdDev = v * np.sqrt(T)
                factor = S0 * np.exp( ( ( r - q ) - 0.5 * v * v ) * T )
                k = K / factor
                if optionType == 'c':
                    f_payoff = lambda x: np.maximum(x - k, 0)
                elif optionType == 'p':
                    f_payoff = lambda x: np.maximum(k - x, 0) 
                else: 
                    raise(ValueError('option type should be call or put.'))
                S = np.exp( logStdDev * dw )
                self.S = S * factor
                payoff = f_payoff(S)
                payoff = np.exp( -r * T ) * factor * payoff
                return np.average(payoff)
        
        if optionType == 'c':
            f_payoff = lambda x: np.maximum(x - K, 0)
        elif optionType == 'p':
            f_payoff = lambda x: np.maximum(K - x, 0) 
        else: 
            raise(ValueError('option type should be call or put.'))

        self.generate_S(n_paths, n_steps, antiVar=True)
        S = self.S

        if European:
            payoff = f_payoff(S[:, -1])
            payoff = np.exp(-r * T) * payoff
            return np.average(payoff)
        if not European:
            return self.LS(f_payoff, k=2)
