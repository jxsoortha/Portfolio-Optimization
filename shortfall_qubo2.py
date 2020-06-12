import numpy as np
import sys
import csv
from scipy import special
import dimod
import neal
import bottleneck as bn
from tabu import TabuSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite


import seaborn as sns
import matplotlib.pyplot as plt
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

def build_qubo(theta_1, theta_2 ,theta_3, n_assets,n_digits,sigma,mu,est,p):
    linear_terms = {}
    quadratic_terms = {}

    #first term
    for i in range(n_assets):
        for j in range(n_digits):
            linear_terms[n_digits*i+j] = (0.5**(j+1))**2-2*(0.5**(j+1))
            linear_terms[n_digits*i+j] *= theta_1

    for i in range(n_assets*n_digits-1):
        for j in range(i+1,n_assets*n_digits):
            d_1 = i % n_digits
            d_2 = j % n_digits
            quadratic_terms[(i,j)] = 2*(0.5**(d_1+1))*(0.5**(d_2+1))
            quadratic_terms[(i,j)] *= theta_1

    #second term
    for i in range(n_assets):
        for j in range(n_digits):  
            linear_terms[n_digits*i+j] += (0.5**(2*j+2)) * ((mu[i])**2) * theta_2
            linear_terms[n_digits*i+j] -= 2*p * (0.5**(j+1)) * mu[i] * theta_2
    
    for i in range(n_assets):
        for j in range(n_assets):
            for l in range(n_digits):
                for k in range(n_digits):
                    if i*n_digits+l >= j*n_digits+k:
                        continue
                    quadratic_terms[(i*n_digits+l,j*n_digits+k)] += 2 * mu[i] * mu[j] * (0.5**(l+k+2)) *theta_2

    #third term
    for i in range(n_assets):
        for j in range(n_digits):
            linear_terms[n_digits*i+j] += (0.5**(2*j+2)) * sigma[i,i] * theta_3

    for i in range(n_assets):
        for j in range(n_assets):
            for l in range(n_digits):
                for k in range(n_digits):
                    if i*n_digits+l >= j*n_digits+k:
                        continue
                    quadratic_terms[(i*n_digits+l,j*n_digits+k)] += 2 * sigma[i,j] * (0.5**(l+k+2)) *theta_3
            
    return linear_terms, quadratic_terms

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



T = 32
time_window = 100
n_digits = 7
epi = 0.02
samples = 3000
W = []
for t in range(1,T):
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
    if p < 0:
        p = 0.99 * np.max(mu)
    r = check_trivial_answer(n_assets,mu,EST,return_matrix[:,starting_index:ending_index])
    if r != -1:
        print("Just invest everything to the ", r+1, "-th asset")
        continue

    ite = 1
    
    while ite < 500:
        print('-----iteration ',ite,' -----')
        ite += 1
        ss = 1 / (np.abs(p) * np.abs(p))
        sss = 1 / np.abs(p)
        linear_terms, quadratic_terms = build_qubo(100,10 * ss,sss/5,n_assets,n_digits,C,mu,EST,p)
        bqm = dimod.BinaryQuadraticModel(linear_terms, quadratic_terms, 0, dimod.BINARY)
        #sampler = EmbeddingComposite(DWaveSampler())
        #sampleset = sampler.sample(bqm, num_reads=samples, chain_strength=20)
        solver = neal.SimulatedAnnealingSampler()
        sampleset = solver.sample(bqm, num_reads=samples)

        assignments = sampleset.record.sample
        energy = sampleset.record.energy

        sorted_assignments = [x for (y,x) in sorted(zip(energy,assignments), key=lambda pair: pair[0])] 
        energy.sort()

        # for k in range(len(sorted_assignments)):
        #     w = np.zeros(n_assets)
        #     for j in range(n_assets):
        #         for m in range(n_digits):
        #             w[j] += sorted_assignments[k][j*n_digits+m]*(0.5**(m+1))
            
        #     for j in range(len(w)):
        #         w[j] = w[j] / np.sum(w)
        #     print("target return is: ", p)
        #     print("computed profit is:", w.T @ mu)
        #     print("volatility is: ", (0.5 * w.T @ C @ w)[0,0])
        #     print("the weights are:", w)
        #     print("---------------------")
        #     print("---------------------")
        # exit()
            

        w = np.zeros(n_assets)
        for j in range(n_assets):
            for m in range(n_digits):
                w[j] += sorted_assignments[0][j*n_digits+m]*(0.5**(m+1))

        

        for j in range(len(w)):
            w[j] = w[j] / np.sum(w)
        print("target return is: ", p)
        print("computed profit is:", w.T @ mu)
        print("volatility is: ", (0.5 * w.T @ C @ w)[0,0])
        print("the weights are:", w)
        w_v = w @ return_matrix[:,starting_index:ending_index]
        length = int(np.floor(alpha*len(w_v)))-1
        ESt = bn.partition(w_v, kth = length)
        ESt = np.mean(ESt[0:length])
        print("ES ratio: ", ESt/EST)
        if ESt/EST > 1 + epi:
            if p >= 0:
                p -= 0.02 * p
            else:
                p += 0.02 * p
        elif ESt/EST < 1 - epi:
            if p >= 0:
                p += 0.02 * p
            else:
                p -= 0.02 * p
        else:
            W.append(w)
            return_std = np.std(w @ return_matrix[:,starting_index:ending_index] )
            print("ES is: ", ESt)
            print("Targer ES is: ", EST)
            print("Sharpe ratio is: ", w.T @ mu / return_std)
            print("Terminated...")
            print()
            print()
            break


print(W)
    # 
    # for i in range(samples):
    #     for j in range(n_assets):
    #         for m in range(n_digits):
    #             w[j] += sorted_assignments[i][j*n_digits+m]*(0.5**(m+1))
    #     print("w:", w, np.sum(w))
    #     print("The optimal value is:", w.T @ mu)
    #     print("---------------")
    #     w = np.zeros(n_assets)
    #     # if np.abs(np.sum(w)) >= 0.9:
    #     #     w /= np.sum(w)
    #     #     break
    # print("---------------")