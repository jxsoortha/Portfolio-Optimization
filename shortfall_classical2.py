import cvxpy as cp
import numpy as np
import pandas as pd
import bottleneck as bn
import csv
from scipy import special


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def read_data(datasheet):
    fields = [] 
    data = {}
    with open(datasheet,'r') as csvfile:
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader) 
        for row in csvreader:
            data[row[0]] = row[4]
        return data

def process_data(data):
    assets = []
    for i in range(len(data)):
        assets.append([])
    asset1 = data[0]
    dates = list(asset1.keys())
    for i in range(len(asset1)):
        all_exist = 1
        for j in range(len(data)-1):
            if dates[i] not in data[j+1]:
                all_exist = 0
        if all_exist == 1:
            for j in range(len(data)):
                assets[j].append(data[j][dates[i]])
    return assets

def check_trivial_answer(n_assets, mu, EST, return_matrix):
    max_index = np.argmax(mu)
    w = np.zeros(n_assets)
    for i in range(n_assets):
        if i != max_index:
            w[i] = 0
        else:
            w[i] = 1

    w_v = w @ return_matrix
    length = int(np.floor(alpha*len(w_v)))-1
    ESt = bn.partition(w_v, kth = length)
    ESt = np.mean(ESt[0:length])

    if (ESt/EST < 1):
        return max_index
    else:
        return -1

alpha = 0.05

gld_data = read_data('GLD.csv')
agg_data = read_data('AGG.csv')
spy_data = read_data('SPY.csv')
eem_data = read_data('EEM.csv')
dbc_data = read_data('DBC.csv')
dbv_data = read_data('DBV.csv')

spy_2008_data = read_data('SPY_2008.csv')
spy_2008_closing = list(spy_2008_data.values())


spy_2008_return = np.zeros(len(spy_2008_closing)-1)
for i in range(len(spy_2008_closing)-1):
    spy_2008_return[i] = (float(spy_2008_closing[i+1])- float(spy_2008_closing[i])) / float(spy_2008_closing[i])

sigma_2008 = np.std(spy_2008_return)
VaR = int(np.floor(alpha * len(spy_2008_return)))
EST_2008 = bn.partition(spy_2008_return, kth = VaR-1)
EST_2008 = np.mean(EST_2008[0:VaR-1])





list_of_data = [spy_data,agg_data,gld_data,eem_data,dbc_data,dbv_data]
#list_of_data = [agg_data,eem_data]
assets = process_data(list_of_data)
assets = np.array(assets)
assets = assets.astype(np.float)
n_assets = assets.shape[0]
n_obs = assets.shape[1]

return_matrix = np.zeros((n_assets,n_obs-1))
for i in range(n_assets):
    for j in range(n_obs-1):
        return_matrix[i,j] = (assets[i][j+1] - assets[i][j]) / assets[i][j]


T = 1
time_window = 100

epi = 0.05

solution_set = []

for t in range(T):
    t = 15
    print("-----------------At T = ", t,"-----------------")
    starting_index = return_matrix.shape[1] - (t+1) * time_window
    ending_index = return_matrix.shape[1] - t * time_window 
    mu = np.zeros(n_assets)
    for i in range(n_assets):
        mu[i] = np.mean(return_matrix[i,starting_index:ending_index])
    sigma_spy_t = np.std(return_matrix[0,starting_index:ending_index])
    EST = sigma_spy_t/sigma_2008 * EST_2008
    C = np.asmatrix(np.cov(return_matrix[:,starting_index:ending_index]))
    p = np.mean(mu)
    r = check_trivial_answer(n_assets,mu,EST,return_matrix[:,starting_index:ending_index])
    if r != -1:
        print("Just invest everything to the ", r+1, "-th asset")
        break
    ite = 1

    while ite < 200:
        print('-----iteration ',ite,' -----')
        ite += 1
        w = cp.Variable(n_assets)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(w,C)),[cp.sum(w) == 1, w >= 0, mu.T @ w == p])
        prob.solve()
        w = w.value
        print("target profit is: ", p)
        print("the weights are:", w)
        w_v = w @ return_matrix[:,starting_index:ending_index]
        length = int(np.floor(alpha*len(w_v)))-1
        ESt = bn.partition(w_v, kth = length)
        ESt = np.mean(ESt[0:length])
        print("ES ratio: ", ESt/EST)
        if ESt/EST > 1 + epi:
            if p >= 0:
                p *= 0.98
            else:
                p *= 1.02
        elif ESt/EST < 1 - epi:
            if p < 0:
                p *= 0.98
            else:
                p *= 1.02
        else:
            
            print("Terminated...")
            print()
            print()
            break







