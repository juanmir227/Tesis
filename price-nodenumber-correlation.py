import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
import os
import pandas as pd
from scipy.signal import savgol_filter

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo1_lists\periodo1_dates', 'rb') as file:
    periodo1_dates = pd.read_pickle(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo2_lists\periodo2_dates', 'rb') as file:
    periodo2_dates = pd.read_pickle(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_lists\periodo3_dates', 'rb') as file:
    periodo3_dates = pd.read_pickle(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\gen_list\number_of_nodes', 'rb') as file:
    number_of_nodes = pd.read_pickle(file)

dates = periodo1_dates + periodo2_dates + periodo3_dates
mean_price = []
price_data = pd.read_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Algo_price_data\algo_price_data.csv')
for date in dates:
    date_row = price_data[price_data['Date'] == str(date)]
    date_mean_price = (float(date_row['Open']) + float(date_row['Close']))/2
    mean_price.append(date_mean_price)
mean_price = savgol_filter(mean_price, 15, 7)
number_of_nodes = savgol_filter(number_of_nodes, 15 ,7)
corr_coef = np.corrcoef(mean_price, number_of_nodes)[0,1]
corr_coef = np.round(corr_coef,2)

print(len(periodo1_dates))
print(len(periodo2_dates))
print(len(periodo3_dates))

periodo1_mean_price = mean_price[:73]
periodo2_mean_price = mean_price[73:126]
periodo3_mean_price = mean_price[126:235]
periodo1_number_of_nodes = number_of_nodes[:73]
periodo2_number_of_nodes = number_of_nodes[73:126]
periodo3_number_of_nodes = number_of_nodes[126:235]

end_mean_price = mean_price/np.max(mean_price)
end_number_of_nodes = number_of_nodes/np.max(number_of_nodes)
data_data_frame= pd.DataFrame({'meanprice': end_mean_price, 'numberofnodes': end_number_of_nodes, 'dates': dates})
csv_data = data_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\price_vs_node_number\price_vs_node_number.csv', index = False)


# print(np.corrcoef(periodo3_mean_price, periodo3_number_of_nodes))

plt.rc('figure', figsize= (15,6))
sb.set_style('darkgrid')
plt.plot(dates,mean_price/np.max(mean_price), label = 'Precio')
plt.plot(dates,number_of_nodes/np.max(number_of_nodes), label = 'Número de nodos')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Porcentaje relativo al máximo')
plt.title('Correlación entre número de nodos y precio del Algo')
plt.show()