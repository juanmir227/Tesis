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
import seaborn as sb
def calc_chisquare(meas, sigma, fit):
    diff = pow(meas-fit, 2.)
    test_statistic = (diff / pow(sigma, 2.)).sum()

    return test_statistic

plt.rc('figure', figsize= (12,5))
sb.set_style('darkgrid')
r2_free = []
r2_exp = []
chi2_free = []
chi2_exp = []
p_values_powerlaw = []
p_values_stretchedexp = []

path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\big_chunk_periodo3\big_chunk_periodo3_dataframe_filtered"
df = pd.read_pickle(path)
G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
temp = list(G.degree())
degree_list = []
for element in temp:
    degree_list.append(element[1])
plt.clf()
n, bins, patches = plt.hist(degree_list, bins = np.arange(1,50,1)-0.5, range = (1,21))
print(np.max(bins)+0.5)
plt.xlabel('Valor de k')
plt.ylabel('Número de cuentas')
plt.title('Distribución de k en período 4')
plt.xticks(list(range(1,int(np.max(bins)+0.5))))
plt.xlim(0,11)
plt.show()
bins = bins[:-1]+0.5
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


# threshold = 10
# bins = bins[:threshold]
# values = values[:threshold]
# density_error = density_error[:threshold]

def model(x,A,gamma):
    return A*x**-gamma
#Aca genero el ajuste por el modelo model
params, pcov = curve_fit(model, xdata =bins, ydata = values, p0 = (values[0],-3), sigma = density_error, maxfev = 50000)
perr_free = np.sqrt(np.diag(pcov))
print(perr_free)
print(f'Gamma value:{params[1]}')
prediction = model(bins, params[0], params[1])

bara_bins = list(range(1,int(bins[-1])+1))
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
try:
    stretch_exp, pcov = curve_fit(stretched_exponential, xdata = bins, ydata = values, p0= (values[0],0.5, 1), sigma = density_error, maxfev = 50000)
    perr_exp = np.sqrt(np.diag(pcov))
    print(f'Alfa Value:{stretch_exp[1]}')
    stretched_values = stretched_exponential(bins, stretch_exp[0], stretch_exp[1], stretch_exp[2])

    
    plt.errorbar(bins,values,yerr = density_error, label = 'data')
    plt.plot(bins, prediction, label = f'Serie de potencias: Gamma = {np.round(params[1],2)} ± {np.round(perr_free[1],2)}',)
    plt.plot(bins,stretched_values, label = f'Exponencial estirada: Alpha = {np.round(stretch_exp[1],2)} ± {np.round(perr_exp[1],2)}')
    plt.plot(bara_bins, bara_values, label = 'barabasi')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.title('Resultados de los ajustes')
    plt.xlabel('Valor de k')
    plt.ylabel('Número de cuentas')
    # plt.xlim(0,100)
    plt.show()


    real_data_frame = pd.DataFrame({'data': values, 'bins': bins, 'error':density_error})
    barabasi_data_frame = pd.DataFrame({'bins': bara_bins, 'barabasi_data': bara_values})
    #Gamma
    gamma_data_frame = pd.DataFrame({'gamma':params[1], 'gamma_err':perr_free[1]}, index = [0])
    potencias_data_frame = pd.DataFrame({'data_potencias': prediction, 'bins_potencias':bins})
    #Alpha
    alpha_data_frame = pd.DataFrame({'alpha': stretch_exp[1], 'alpha_err': perr_exp[1]}, index = [0])
    exponencial_data_frame = pd.DataFrame({'data_stretched_exp': stretched_values, 'bins_stretched_exp':bins})

    barabasi_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\big_periods_data\periodo3_barabasi_data.csv', index = False)
    real_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\big_periods_data\periodo3_real_data.csv', index = False)
    gamma_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\big_periods_data\periodo3_gamma_data.csv', index = False)
    potencias_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\big_periods_data\periodo3_serie_potencias_data.csv', index = False)
    alpha_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\big_periods_data\periodo3_alpha_data.csv', index = False)
    exponencial_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\big_periods_data\periodo3_stretched_exp_data.csv', index = False)


    r_2_power_law = r2_score(values,prediction)
    r_2_stretched_exp = r2_score(values, stretched_values)
    # print(f'R_2 Power Law:{r_2_power_law}')
    # print(f'R_2 Stretched Exponential:{r_2_stretched_exp}')
    p_value_power_law = ttest_ind(values, prediction).pvalue
    p_value_stretched_exp = ttest_ind(values, stretched_values).pvalue
    p_values_powerlaw.append(p_value_power_law)
    p_values_stretchedexp.append(p_value_stretched_exp)
    TS_free = calc_chisquare(values, density_error, prediction) #df = 20 - 2 = 18
    TS_exp = calc_chisquare(values, density_error, stretched_values) #df = 20 - 3 = 17
    # print(p_value_power_law)
    # print(p_value_stretched_exp)
    # print(TS_free)
    # print(TS_exp)
    r2_free.append(r_2_power_law)
    r2_exp.append(r_2_stretched_exp)
    chi2_free.append(TS_free)
    chi2_exp.append(TS_exp)
except RuntimeError:
    pass

print(r2_exp)
print(r2_free)

# plt.clf()
# plt.plot(r2_free, label = 'Power Law')
# plt.plot(r2_exp, label = 'Exponential')
# plt.legend()
# plt.title('R2')
# plt.xlabel('')
# plt.ylabel('R2 Value')
# plt.show()

# plt.plot(chi2_free, label = 'Power Law')
# plt.plot(chi2_exp,label = 'Exponential')
# plt.legend()
# plt.title('Chi Square')
# plt.xlabel('')
# plt.ylabel('ChiSquare Value')
# plt.show()


# plt.plot(p_values_powerlaw, label = 'Power Law')
# plt.plot(p_values_stretchedexp, label = 'Exponential')
# plt.legend()
# plt.title('P_values')
# plt.xlabel('')
# plt.ylabel('P Value')
# plt.show()