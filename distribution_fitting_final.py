from algosdk.v2client import algod
import pickle
from getBlocks import GetBlockInfo
from acquireTxns import join_txns
from txnsDataframe import make_df
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import chisquare
from scipy.stats import ttest_ind

def calc_chisquare(meas, sigma, fit):
    diff = pow(meas-fit, 2.)
    test_statistic = (diff / pow(sigma, 2.)).sum()

    return test_statistic

r2_free = []
r2_exp = []
chi2_free = []
chi2_exp = []
p_values_powerlaw = []
p_values_stretchedexp = []

initial_block1 = 11000000
final_block1 = 14600000
initial_block2 = 14800000
final_block2=17400000
initial_block3 =17600000
final_block3 = 23000000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block1, final_block1 + increment, increment))+list(range(initial_block2, final_block2 + increment, increment)) +list(range(initial_block3, final_block3 + increment, increment))
# 
for init_number in initial_number:
    print(init_number)
    if init_number <= 14600000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo1_dataframe\periodo1_"
    elif 14800000<=init_number<=17400000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo2_dataframe\periodo2_"
    elif init_number >= 17600000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_dataframe\periodo3_"
    df = pd.read_pickle(path+str(init_number)+"_"+str(number_of_blocks))

    G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
    temp = list(G.degree())
    degree_list = []
    for element in temp:
        degree_list.append(element[1])
    plt.clf()
    n, bins, patches = plt.hist(degree_list, bins = 20, range = (1,21))
    # plt.show()
    bins = bins[:-1]
    error = np.sqrt(n)

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


    threshold = 10
    bins = bins[:threshold]
    values = values[:threshold]
    density_error = density_error[:threshold]

    def model(x,gamma):
        return values[0]*x**gamma
    #Aca genero el ajuste por el modelo model
    params, pcov = curve_fit(model, xdata =bins, ydata = values, p0 = (-3), sigma = density_error, maxfev = 50000)
    print(f'Gamma value:{params[0]}')
    prediction = model(bins, params[0])
  
    bara_bins = list(range(1,int(bins[-1])+1))
    # print(bara_bins)
    A = values[0]
    exponent = -3
    bara_values = [A*x**exponent for x in bara_bins]

    def fit_poisson(k,mu):
        return values[0]*poisson.pmf(k,mu)

    poisson_params, pcov = curve_fit(fit_poisson, xdata = bins, ydata = values, sigma = density_error)
    print(poisson_params)
    plt.errorbar(bins,values,yerr=density_error, label = 'Poisson')

    # print(values[0])
    mean_k = np.mean(np.array(degree_list))
    def stretched_exponential(x,alfa, mu):
        mean_x = mean_k
        return values[0]*x**(-alfa)*np.exp((-2*mu)/(mean_x*(1-alfa))*x**(1-alfa))
    #aca genero el ajuste por el modelo stretched_eponential
    try:
        stretch_exp, pcov = curve_fit(stretched_exponential, xdata = bins, ydata = values, p0= (0.5, 1), sigma = density_error, maxfev = 50000)
        print(f'Alfa Value:{stretch_exp[1]}')
        stretched_values = stretched_exponential(bins, stretch_exp[0], stretch_exp[1])
        plt.errorbar(bins,values,yerr = density_error, label = 'data')
        plt.plot(bins, prediction, label = 'fit')
        plt.plot(bins,stretched_values, label = 'stretched exp')
        plt.plot(bara_bins, bara_values, label = 'barabasi')

        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.title('Ajustes')
        # plt.xlim(0,100)
        if init_number == 14300000 or init_number == 17000000 or init_number == 212000000:
            plt.show()

        r_2_power_law = r2_score(values,prediction)
        r_2_stretched_exp = r2_score(values, stretched_values)
        print(f'R_2 Power Law:{r_2_power_law}')
        print(f'R_2 Stretched Exponential:{r_2_stretched_exp}')
        p_value_power_law = ttest_ind(values, prediction).pvalue
        p_value_stretched_exp = ttest_ind(values, stretched_values).pvalue
        p_values_powerlaw.append(p_value_power_law)
        p_values_stretchedexp.append(p_value_stretched_exp)
        TS_free = calc_chisquare(values, density_error, prediction) #df = 20 - 2 = 18
        TS_exp = calc_chisquare(values, density_error, stretched_values) #df = 20 - 3 = 17
        print(p_value_power_law)
        print(p_value_stretched_exp)
        print(TS_free)
        print(TS_exp)
        r2_free.append(r_2_power_law)
        r2_exp.append(r_2_stretched_exp)
        chi2_free.append(TS_free)
        chi2_exp.append(TS_exp)
    except RuntimeError:
        pass



plt.clf()
plt.plot(r2_free, label = 'Power Law')
plt.plot(r2_exp, label = 'Exponential')
plt.legend()
plt.title('R2')
plt.xlabel('')
plt.ylabel('R2 Value')
plt.show()

plt.plot(chi2_free, label = 'Power Law')
plt.plot(chi2_exp,label = 'Exponential')
plt.legend()
plt.title('Chi Square')
plt.xlabel('')
plt.ylabel('ChiSquare Value')
plt.show()


plt.plot(p_values_powerlaw, label = 'Power Law')
plt.plot(p_values_stretchedexp, label = 'Exponential')
plt.legend()
plt.title('P_values')
plt.xlabel('')
plt.ylabel('P Value')
plt.show()