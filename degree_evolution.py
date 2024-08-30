import pickle
import os
import sys
from getBlocks import GetBlockInfo
from acquireTxns import join_txns
from txnsDataframe import make_df
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques

initial_block = 11000000
final_block = 18000000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block, final_block + increment, increment))
mean_degree_evolution = []
graphs = []
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_dataframe\periodo3_'+ str(final_block)+"_"+str(number_of_blocks), 'rb') as fp:
    df = pickle.load(fp)
G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
graphs.append(G)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_dataframe\periodo3_'+ str(22800000)+"_"+str(number_of_blocks), 'rb') as fp:
    data = pickle.load(fp)
D = nx.from_pandas_edgelist(data,'Sender Address', 'Receiver Address')
graphs.append(D)

# temp = list(G.degree())
# degree_list = []
# for element in temp:
#     degree_list.append(element[1])
print(list(G.degree())[0])

