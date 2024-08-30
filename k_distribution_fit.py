from algosdk.v2client import algod
import pickle
from get_blocks import GetBlockInfo
from acquire_txns import join_txns
from txns_dataframe import make_df
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import chisquare

#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques

initial_block = 11000000
final_block = 19000000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block, final_block + increment, increment))
mean_degree_evolution = []

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_dataframe\periodo3_'+ str(final_block)+"_"+str(number_of_blocks), 'rb') as fp:
    df = pickle.load(fp)

G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
temp = list(G.degree())
degree_list = []
for element in temp:
    degree_list.append(element[1])

n, bins, patches = plt.hist(degree_list, bins = 20, range = (1,21))
plt.show()
plt.close()
bins = bins[:-1]
error = np.sqrt(n)
# print(n, bins)

# values = n/np.sum(n)
values = n
# print(bins)
# density_error = error/np.sum(n)
density_error = error
indexes = []
new_values = []
new_bins = []
new_density_error = []
# print(values)

for i,el in enumerate(values):
    if el != 0:
        new_values.append(values[i])
        new_bins.append(bins[i])
        new_density_error.append(density_error[i])

bins = new_bins
values = new_values
density_error = np.array(new_density_error)


# threshold = 10
# bins = bins[:threshold]
# values = values[:threshold]
# density_error = density_error[:threshold]

def model(x,A,gamma):
    return A*x**gamma
#Aca genero el ajuste por el modelo model
params, pcov = curve_fit(model, xdata =bins, ydata = values, p0 = (0.6,-3), sigma = density_error)
print(f'Gamma value:{params[1]}')
prediction = model(bins, params[0], params[1])
bara_bins = list(range(1,bins[-1]))
# print(bara_bins)
A = values[0]
exponent = -3
bara_values = [A*x**exponent for x in bara_bins]

# print(values[0])
mean_k = np.mean(np.array(degree_list))
def stretched_exponential(x, A,alfa, mu):
    mean_x = mean_k
    return A*x**(-alfa)*np.exp((-2*mu)/(mean_x*(1-alfa))*x**(1-alfa))
#aca genero el ajuste por el modelo stretched_eponential
stretch_exp, pcov = curve_fit(stretched_exponential, xdata = bins, ydata = values, p0= (0.7,0.5, 1), sigma = density_error)
print(f'Alfa Value:{stretch_exp[1]}')
stretched_values = stretched_exponential(bins, stretch_exp[0], stretch_exp[1], stretch_exp[2])
plt.errorbar(bins,values,yerr = density_error, label = 'data')
plt.plot(bins, prediction, label = 'fit')
plt.plot(bins,stretched_values, label = 'stretched exp')
plt.plot(bara_bins, bara_values, label = 'barabasi')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('Ajustes')
# plt.xlim(0,100)
plt.show()

print(f'R_2 Power Law:{r2_score(values,prediction)}')
print(f'R_2 Stretched Exponential:{r2_score(values, stretched_values)}')


# G_barabasi = nx.barabasi_albert_graph(2000,1)
# temp_barabasi = list(G_barabasi.degree())
# degree_list_barabasi = []
# for element in temp_barabasi:
#     degree_list_barabasi.append(element[1])
# n_barbasi, bins_barbasi, patches = plt.hist(degree_list_barabasi, bins = 20, range = (1,21))
# plt.show()

# bins_barbasi = bins_barbasi[:-1]
# error_barabasi = np.sqrt(n_barbasi)
# print(n_barbasi,bins_barbasi)
# plt.plot(bins, n, label = 'data')
# plt.plot(bins,n_barbasi, label = 'simulation')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.show()

#voy a calcular el chisquared
# print(np.sum(values))
# print(np.sum(prediction))
# print(np.sum(stretched_values))
# print(n_barbasi)
# # grados de libertad = n-1 = 19

# crit = stats.chi2.ppf(q = 0.95, df = 19)

# print(f'Critical Value: {crit}')

# # p_value = 1 - stats.chi2.cdf(x = chi_squared_stat, df = 19)

# print(chisquare(f_obs = values, f_exp = prediction))




def calc_chisquare(meas, sigma, fit):
    diff = pow(meas-fit, 2.)
    test_statistic = (diff / pow(sigma, 2.)).sum()

    return test_statistic

TS_free = calc_chisquare(values, density_error, prediction) #df = 20 - 2 = 18
TS_exp = calc_chisquare(values, density_error, stretched_values) #df = 20 - 3 = 17
print(TS_free)
print(TS_exp)
