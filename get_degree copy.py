from algosdk.v2client import algod
import pickle
import os
from dotenv import load_dotenv
load_dotenv()
from get_blocks import GetBlockInfo
from acquire_txns import join_txns
from txns_dataframe import make_df
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques

initial_block = 11000000
final_block = 14600000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block, final_block + increment, increment))
mean_degree_evolution = []


for init_block in initial_number:
    with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Try_periods\periodo1_data_frame\df_'+ str(init_block)+"_"+str(number_of_blocks)+'_filtered', 'rb') as fp:
        df = pickle.load(fp)
    # print(len(df['Sender Address'].tolist()))
    G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
    temp = list(G.degree())
    degree_list = []
    for element in temp:
        degree_list.append(element[1])

    # print(degree_list)
    mean_degree_evolution.append(np.mean(np.array(degree_list)))
    plt.hist(degree_list, bins = 21, range=(0,20), density = True)
    plt.xlabel('Degree(K)')
    plt.ylabel('Probability (Pk)')
    plt.show()
# print(mean_degree_evolution)
# plt.plot(mean_degree_evolution)
# plt.show()