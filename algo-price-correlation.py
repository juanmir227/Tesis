import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
import os
import pandas as pd
from scipy.signal import savgol_filter



price_data = pd.read_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Algo_price_data\algo_price_data.csv')

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_dataframe\periodo3_17600000_500', 'rb') as file:
    data = pd.read_pickle(file)

pay_data = data[data['Transaction Type'] == 'pay']
amount = pay_data['Transaction Amount'].tolist()
amount = [float(x) for x in amount if x != 'NA']
multiplier = 0.000001
algo_amount = []
algo_amount = [round(element*multiplier,2) for element in amount]
total_amount = round(sum(algo_amount,2))
#aca ya resolvi cuanto algo se envio en este chunk, ahora defino una fecha
dates = data["Transaction Date"].tolist()
new_dates = [date.date() for date in dates]
chunk_date = new_dates[0]
date_row = price_data[price_data['Date'] == str(chunk_date)]
chunk_mean_price = (float(date_row['Open']) + float(date_row['Close']))/2
print(chunk_date, chunk_mean_price, total_amount)

#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques

initial_block = 17600000
final_block = 23000000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block, final_block + increment, increment))
path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_dataframe\periodo3_"
chunk_dates = []
chunk_mean_prices = []
total_amounts = []

for init_number in initial_number:
    data = pd.read_pickle(path+str(init_number)+"_"+str(number_of_blocks))
    pay_data = data[data['Transaction Type'] == 'pay']
    amount = pay_data['Transaction Amount'].tolist()
    amount = [float(x) for x in amount if x != 'NA']
    multiplier = 0.000001
    algo_amount = []
    algo_amount = [round(element*multiplier,2) for element in amount]
    total_amount = round(sum(algo_amount,2))
    #aca ya resolvi cuanto algo se envio en este chunk, ahora defino una fecha
    dates = data["Transaction Date"].tolist()
    new_dates = [date.date() for date in dates]
    chunk_date = new_dates[0]
    date_row = price_data[price_data['Date'] == str(chunk_date)]
    chunk_mean_price = (float(date_row['Open']) + float(date_row['Close']))/2
    chunk_dates.append(chunk_date)
    chunk_mean_prices.append(chunk_mean_price)
    total_amounts.append(total_amount)

# print(chunk_dates)
# print(chunk_mean_prices)
# print(total_amounts)

print(np.corrcoef(chunk_mean_prices, total_amounts)[0,1])

# print(np.max(np.array(chunk_mean_prices)))
total_amounts = savgol_filter(total_amounts, 25, 7)

try_prices = [x/np.max(np.array(chunk_mean_prices)) for x in chunk_mean_prices]
try_amounts = [x/np.max(np.array(total_amounts)) for x in total_amounts]



plt.rc('figure', figsize= (15,6))
sb.set_style('darkgrid')
plt.plot(chunk_dates, try_prices, label = 'Precio')
plt.plot(chunk_dates, try_amounts, label = 'Cantidad de Algos')
plt.legend()
plt.title('Correlación entre cantidad de Algos por chunk y su precio')
plt.xlabel('Fecha')
plt.ylabel('Porcentaje relativo al máximo')
plt.show()
