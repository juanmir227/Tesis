from algosdk.v2client import algod
import pickle
import os
import sys
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
from scipy.stats import kstest

#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques

initial_block = 11000000
final_block = 19400000
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
bins = bins[:-1]
error = np.sqrt(n)
# print(n, bins)

values = n/np.sum(n)
print(bins)
density_error = error/np.sum(n)
indexes = []
new_values = []
new_bins = []
new_density_error = []
print(values)
for i,el in enumerate(values):
    if el != 0:
        new_values.append(values[i])
        new_bins.append(bins[i])
        new_density_error.append(density_error[i])

bins = new_bins
values = new_values
density_error = new_density_error

# for i in range(len(values)):
#     if values[i] == 0:
#         values[i] = 1e-8
# for i in range(len(bins)):
#         if bins[i] == 0:
#             bins[i] = 1e-8
# for i in range(len(density_error)):
#         if density_error[i] == 0:
#             density_error[i] = 1e-8
# print(values, density_error)
# for i in range(len(n)):
#     if n[i] == 0:
#         n[i] = 1e-8
# for i in range(len(bins)):
#         if bins[i] == 0:
#             bins[i] = 1e-8
# for i in range(len(error)):
#         if error[i] == 0:
#             error[i] = 1e-8

def model(x,a,gamma):
    return a*x**gamma

# threshold = 10
# bins = bins[:threshold]
# values = values[:threshold]
# density_error = density_error[:threshold]

params, pcov = curve_fit(model, xdata =bins, ydata = values, p0 = (0.47, -1), sigma = density_error)
print(params)
plt.errorbar(bins,values,yerr = density_error, label = 'data')
prediction = model(bins, params[0], params[1])
plt.plot(bins, prediction, label = 'fit')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Ajuste por X^gamma')
plt.show()

# def new_model(x,a,gamma):
#      return a*np.exp(x*gamma)

# new_params,pcov = curve_fit(new_model,xdata =bins, ydata = values, p0 = (1.3, -1), sigma = density_error)
# print(new_params)
# plt.errorbar(bins,values,yerr = density_error, label = 'data')
# plt.plot(bins, new_model(bins, new_params[0], new_params[1]), label = 'fit')
# plt.title('Ajuste por e^(x*gamma)')
# # plt.plot(bins, new_model(bins, 1.5, -0.7), color = 'pink')
# plt.show()

# #Voy a implementar una distribucion de poisson
# def fit_poisson(k,mu):
#      return poisson.pmf(k-1,mu)

# poisson_params, pcov = curve_fit(fit_poisson, xdata = bins, ydata = values, sigma = density_error)
# print(poisson_params)
# plt.errorbar(bins,values,yerr=density_error, label = 'data')
# plt.plot(bins, fit_poisson(bins, poisson_params[0]), label = 'fit')
# plt.title('Ajuste por Poisson')
# plt.show()

mean_k = np.mean(np.array(degree_list))
def my_distro(x, A, alfa, mu):
    mean_x = mean_k
    return A*x**(-alfa)*np.exp((-2*mu)/(mean_x*(1-alfa))*x**(1-alfa))

stretch_exp, pcov = curve_fit(my_distro, xdata = bins, ydata = values, p0= (0.7,0.5, 1), sigma = density_error)
plt.errorbar(bins,values,yerr=density_error, label = 'data')
y = my_distro(bins, stretch_exp[0], stretch_exp[1], stretch_exp[2])
print(stretch_exp)
plt.plot(bins,y, label = 'stretched exp')
plt.yscale('log')
plt.xscale('log')
plt.title('Ajuste por Sub Preferential')
plt.show()

print(r2_score(values,prediction))
print(r2_score(values, y))

# G_barabasi = nx.barabasi_albert_graph(970,1)
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