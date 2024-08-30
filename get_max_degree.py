import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
import os
import pandas as pd
import networkx as nx

initial_block1 = 11000000
final_block1 = 14600000
initial_block2 = 14800000
final_block2=17400000
initial_block3 =17600000
final_block3 = 23000000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block1, final_block1 + increment, increment))+list(range(initial_block2, final_block2 + increment, increment)) +list(range(initial_block3, final_block3 + increment, increment))
max_degree = []
chunk_dates = []
chunk_mean_prices = []
total_amounts = []
number_of_nodes = []
for init_number in initial_number:
    if init_number <= 14600000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo1_dataframe\periodo1_"
    elif 14800000<=init_number<=17400000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo2_dataframe\periodo2_"
    elif init_number >= 17600000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_dataframe\periodo3_"
    df = pd.read_pickle(path+str(init_number)+"_"+str(number_of_blocks))
    G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
    number_of_nodes.append(G.number_of_nodes())
    temp = list(G.degree())
    degree_list = []
    for element in temp:
        degree_list.append(element[1])

    max_degree.append(max(degree_list))

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\gen_list\max_degree_list', 'wb') as fp:
    pickle.dump(max_degree, fp)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\gen_list\number_of_nodes', 'wb') as fp:
    pickle.dump(number_of_nodes, fp)