#%%
from MonteCarlo import MonteCarlo
import pandas as pd
import numpy as np
from tqdm import tqdm 

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
                
#%%
# data.to_csv('data.csv', index=False)
# %%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
data = pd.read_csv('data.csv')
x = data[['moneyness', 'tau', 'r', 'vol']]
y = data['option price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, shuffle=True)
ANN = MLPRegressor(hidden_layer_sizes=(3,), activation='relu', solver='adam', alpha=0.01, batch_size='auto', learning_rate_init=0.05, max_iter=200, shuffle=True, verbose=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ANN.fit(x_train, y_train)
ANN.score(x_test, y_test)

# %%
