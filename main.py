#%%
from MonteCarlo import MonteCarlo
import pandas as pd
import numpy as np
from tqdm import tqdm 

#%% 
# GBM
# cols = ['moneyness', 'tau', 'r', 'vol', 'option price']
# data = pd.DataFrame(columns=cols)
# for S0 in tqdm(np.arange(10, 20+0.01, 1)):
#     for K in np.arange(10, 20+0.01, 1):
#         for tau in np.arange(0.3, 0.95+0.1, 0.1):
#             for r in np.arange(0.03, 0.08+0.01, 0.01):
#                 for vol in np.arange(0.2, 0.9+0.01, 0.1):
#                     moneyness = np.log( S0 / K ) 
#                     MC = MonteCarlo(model='GBM', S0=S0, K=K, T=tau, r=r, q=0, v=vol)
#                     MC.generate_S(1000,10)
#                     option_price = MC.pricer(optionType='c')
#                     data = pd.concat([data, pd.DataFrame([[moneyness, tau, r, vol, option_price]], columns=cols)], axis=0)

# data.to_csv('data_GBM.csv', index=False)

#%%
# Heston
# cols = ['moneyness', 'tau', 'r', 'vol0', 'theta', 'kappa', 'gamma', 'rho', 'option price']
# data = pd.DataFrame(columns=cols)
# for S0 in np.arange(10, 20+0.1, 0.5): # 20
#     for K in tqdm(np.arange(10, 20+1, 0.5)): # 20
#         for tau in np.arange(0.3, 1+0.1, 0.1): # 7
#             for r in np.arange(0.03, 0.03+0.01, 0.01): # 2
#                 for vol0 in np.arange(0.2, 0.2+0.1, 0.1): # 1
#                     for theta in np.arange(0.2, 0.2+0.1, 0.1): # 1
#                         for kappa in np.arange(0.5, 2+0.5, 0.5): # 3
#                             for gamma in np.arange(0.3, 0.3+0.1, 0.1): # 2
#                                 for rho in np.arange(-0.2, -0.2+0.1, 0.1): # 1
#                                     moneyness = np.log( S0 / K ) 
#                                     MC = MonteCarlo(model='Heston', S0=S0, K=K, T=tau, r=r, q=0, v0=vol0, theta=theta, kappa=kappa, gamma=gamma, rho=rho)
#                                     MC.generate_S(1000,10)
#                                     option_price = MC.pricer(optionType='c')
#                                     data = pd.concat([data, pd.DataFrame([[moneyness, tau, r, vol0, theta, kappa, gamma, rho, option_price]], columns=cols)], axis=0)

# data.to_csv('data_Heston.csv', index=False)

# %%

# GBM
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data_GBM.csv')
x = data[['moneyness', 'tau', 'r', 'vol']]
y = data['option price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, shuffle=True)

StdScaler = StandardScaler()
x_train = StdScaler.fit_transform(x_train)

ANN = MLPRegressor(hidden_layer_sizes=(4,), activation='relu', solver='adam', batch_size='auto', learning_rate_init=0.05, max_iter=50, shuffle=True, verbose=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ANN.fit(x_train, y_train)

StdScaler = StandardScaler()
x_test = StdScaler.fit_transform(x_test)

ANN.score(x_test, y_test)

#%% 

# Heston
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data_Heston.csv')
x = data[['moneyness', 'tau', 'r', 'vol0', 'theta', 'kappa', 'gamma', 'rho']]
y = data['option price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None, shuffle=True)

StdScaler = StandardScaler()
x_train = StdScaler.fit_transform(x_train)

ANN = MLPRegressor(hidden_layer_sizes=(4,), activation='relu', solver='adam', batch_size='auto', learning_rate_init=0.1, max_iter=5, shuffle=True, verbose=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ANN.fit(x_train, y_train)

StdScaler = StandardScaler()
x_test = StdScaler.fit_transform(x_test)

ANN.score(x_test, y_test)
# %%
