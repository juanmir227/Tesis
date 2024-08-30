import pickle
from get_blocks import GetBlockInfo
from acquire_txns import join_txns
from txns_dataframe import make_df
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques
init_transaction_amount = []
end_transaction_amount = []
shortest_paths = []
total_blacklisted_addresses = []
data_frames = []
mean_degree_evolution = []
number_of_blocks = 500
increment = 50000
init_1 = 11000000
final_1 = 14600000
init_2 = 14800000
final_2 = 17400000
init_3 = 17600000
final_3 = 23000000
stop = 0
periodo1 = list(range(init_1, final_1 + increment, increment))
periodo2 = list(range(init_2, final_2 + increment, increment))
periodo3 = list(range(init_3, final_3 + increment, increment))
initial_number = periodo1 + periodo2 + periodo3
for init_number in initial_number:

    if init_1<=init_number<=final_1:
        periodo = '1'
    if init_2<=init_number<=final_2:
        periodo = '2'
    if init_3<=init_number<=final_3:
        periodo = '3'
    
    with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Try_Periods\periodo'+periodo+'_data_frame\df__'+ str(init_number)+"_"+str(number_of_blocks), 'rb') as fp:
        df = pickle.load(fp)


    # print(len(df['Sender Address'].tolist()))
    G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
    temp = list(G.degree())
    degree_list = []
    for element in temp:
        degree_list.append(element[1])

    # print(degree_list)
    mean_degree_evolution.append(np.mean(np.array(degree_list)))



with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo1_lists\periodo1_dates', 'rb') as file:
    chunk_dates_1 = pickle.load(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo2_lists\periodo2_dates', 'rb') as file:
    chunk_dates_2 = pickle.load(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_dates', 'rb') as file:
    chunk_dates_3 = pickle.load(file)


chunk_dates = chunk_dates_1 + chunk_dates_2 + chunk_dates_3
filtered_mean_degree = savgol_filter(mean_degree_evolution, 30,7)
plt.plot(chunk_dates,filtered_mean_degree, color = 'red', label = 'filtered')
plt.plot(chunk_dates, mean_degree_evolution, color = 'pink', label = 'data')
plt.xlabel('Date')
plt.ylabel('Mean Degree')
plt.legend()
plt.show()