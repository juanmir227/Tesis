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
from scipy.signal import savgol_filter




def calc_chisquare(meas, sigma, fit):
    diff = pow(meas-fit, 2.)
    test_statistic = (diff / pow(sigma, 2.)).sum()

    return test_statistic

alphas = []
gammas = []
r2_free = []
r2_exp = []
chi2_free = []
chi2_exp = []
p_values_powerlaw = []
p_values_stretchedexp = []

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo1_lists\periodo1_dates', 'rb') as file:
    chunk_dates1 = pickle.load(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo2_lists\periodo2_dates', 'rb') as file:
    chunk_dates2 = pickle.load(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_lists\periodo3_dates', 'rb') as file:
    chunk_dates3 = pickle.load(file)

dates = chunk_dates1 + chunk_dates2 + chunk_dates3

initial_block1 = 11000000
final_block1 = 14600000
initial_block2 = 14800000
final_block2=17400000
initial_block3 =17600000
final_block3 = 23000000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block1, final_block1 + increment, increment))+list(range(initial_block2, final_block2 + increment, increment)) +list(range(initial_block3, final_block3 + increment, increment))

plt.rc('figure', figsize= (12,5))
sb.set_style('darkgrid')

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
    n, bins, patches = plt.hist(degree_list, bins = np.arange(1,50,1)-0.5, range = (1,21))
    print(np.max(bins)+0.5)
    plt.xlabel('Valor de k')
    plt.ylabel('Número de cuentas')
    plt.title('Distribución de k en período 3')
    plt.xticks(list(range(1,int(np.max(bins)+0.5))))
    plt.xlim(0,11)
    # if init_number > 20450000:
    #     plt.show()
    plt.clf()
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


    threshold = 10
    bins = bins[:threshold]
    values = values[:threshold]
    density_error = density_error[:threshold]
    print(bins)
    def model(x,A,gamma):
        return A*x**-gamma
    #Aca genero el ajuste por el modelo de serie de potencias
    params, pcov = curve_fit(model, xdata =bins, ydata = values, p0 = (values[0],-3), sigma = density_error, maxfev = 50000)
    print(f'Gamma value:{params[1]}')
    perr_free = np.sqrt(np.diag(pcov))
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
        print(f'Alfa Value:{stretch_exp[1]}')
        perr_exp = np.sqrt(np.diag(pcov))
        stretched_values = stretched_exponential(bins, stretch_exp[0], stretch_exp[1], stretch_exp[2])

        plt.errorbar(bins,values,yerr = density_error, label = 'Datos originales')
        plt.plot(bins, prediction, label = f'Ley de potencias: {chr(947)} = {np.round(params[1], 2)} ± {np.round(perr_free[1],2)}')
        # plt.plot(bins,stretched_values, label = f'Exponencial estirada: {chr(945)} = {np.round(stretch_exp[1],2)}± {np.round(perr_exp[1],2)}')
        plt.plot(bara_bins, bara_values, label = f'Barabasi-Albert: {chr(947)} = 3')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.title('Resultados de los ajustes')
        plt.xlabel('Valor de k')
        plt.ylabel('Número de cuentas')
        if init_number == 11300000 or init_number == 16450000 or init_number == 20450000:
            plt.show()
            #Guardo el valor de Alpha, gamma y los datos para cada uno de los ajustes
            #Real
            real_data_frame = pd.DataFrame({'data': values, 'bins': bins, 'error':density_error})
            barabasi_data_frame = pd.DataFrame({'bins': bara_bins, 'barabasi_data': bara_values})
            #Gamma
            gamma_data_frame = pd.DataFrame({'gamma':params[1], 'gamma_err':perr_free[1]}, index = [0])
            potencias_data_frame = pd.DataFrame({'data_potencias': prediction, 'bins_potencias':bins})
            #Alpha
            alpha_data_frame = pd.DataFrame({'alpha': stretch_exp[1], 'alpha_err': perr_exp[1]}, index = [0])
            exponencial_data_frame = pd.DataFrame({'data_stretched_exp': stretched_values, 'bins_stretched_exp':bins})
            #periodo1
            if init_number == 11300000 :
                barabasi_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo1_barabasi_data.csv', index = False)
                real_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo1_real_data.csv', index = False)
                gamma_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo1_gamma_data.csv', index = False)
                potencias_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo1_serie_potencias_data.csv', index = False)
                alpha_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo1_alpha_data.csv', index = False)
                exponencial_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo1_stretched_exp_data.csv', index = False)
            
            elif init_number == 16450000:
                barabasi_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo2_barabasi_data.csv', index = False)
                real_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo2_real_data.csv', index = False)
                gamma_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo2_gamma_data.csv', index = False)
                potencias_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo2_serie_potencias_data.csv', index = False)
                alpha_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo2_alpha_data.csv', index = False)
                exponencial_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo2_stretched_exp_data.csv', index = False)
            elif init_number == 20450000:
                barabasi_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo3_barabasi_data.csv', index = False)
                real_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo3_real_data.csv', index = False)
                gamma_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo3_gamma_data.csv', index = False)
                potencias_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo3_serie_potencias_data.csv', index = False)
                alpha_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo3_alpha_data.csv', index = False)
                exponencial_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\periods_data\periodo3_stretched_exp_data.csv', index = False)

        alphas.append(stretch_exp[1])
        gammas.append(params[1])    
        r_2_power_law = r2_score(values,prediction)
        r_2_stretched_exp = r2_score(values, stretched_values)
        print(f'R_2 Power Law:{r_2_power_law}')
        print(f'R_2 Stretched Exponential:{r_2_stretched_exp}')
        r2_free.append(r_2_power_law)
        r2_exp.append(r_2_stretched_exp)

    except RuntimeError:
        pass

temp = 0
for r2 in r2_free:
    if r2 > 0.95:
        temp = temp + 1
print(temp/len(r2_free))

temp = 0
for r2 in r2_exp:
    if r2 > 0.95:
        temp = temp + 1
print(temp/len(r2_exp))


gammas_filtered = savgol_filter(gammas, 50, 7)

gammas_data_frame = pd.DataFrame({'dates': dates, 'gammas': gammas, 'gammas_filtered':gammas_filtered})
gammas_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\gammas_data.csv', index = False)

plt.clf()
plt.plot(dates, gammas, label = 'Datos')
plt.plot(dates, gammas_filtered,color = 'orange', label = 'Tendencia')
plt.legend()
plt.title(f'Valores del parámetro {chr(947)}')
plt.xlabel('Fecha')
plt.ylabel(f'Valor de {chr(947)}')
plt.show()


alphas_cut = alphas[110:]
mean_alpha_cut = np.mean(alphas_cut)
mean_err = np.var(alphas_cut)


alphas_data_frame = pd.DataFrame({'dates': dates[110:], 'alphas': alphas[110:], 'alpha_prom':mean_alpha_cut*np.ones(len(dates[110:]))})
alphas_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\alphas_data.csv', index = False)
mean_alpha_data_frame = pd.DataFrame({'mean_alpha': mean_alpha_cut, 'mean_alpha_error': mean_err}, index = [0])
mean_alpha_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\mean_alpha.csv', index = False)



plt.plot(dates[110:], alphas[110:], label = 'Datos')
plt.errorbar(dates[110:], mean_alpha_cut*np.ones(len(dates[110:])), label = f'Promedio: {chr(945)} = {np.round(mean_alpha_cut,2)} ± {np.round(mean_err,2)}')
plt.legend()
plt.title(f'Valores del parámetro {chr(945)}')
plt.xlabel('Fecha')
plt.ylabel(f'Valor de {chr(945)}')
plt.show()


r2_data_frame = pd.DataFrame({'dates': dates[110:], 'r2_free': r2_free[110:], 'r2_exp':r2_exp[110:]})
r2_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\distribution_fitting\r2_data.csv', index = False)


plt.clf()
plt.plot(dates[110:], r2_free[110:], label = 'Ley de potencias')
plt.plot(dates[110:],r2_exp[110:], label = 'Exponencial estirada')
plt.legend()
plt.title('R2')
plt.xlabel('Fecha')
plt.ylabel('Valor de R2')
plt.show()