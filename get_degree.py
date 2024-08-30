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

#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques



initial_block = 20000000
final_block = 2020000
number_of_blocks = 20000
increment = 1
initial_number = list(range(initial_block, final_block + increment, increment))
initial_block_number = 22600000
number_of_blocks = 500


with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_data_frame\df_'+ str(initial_block_number)+"_"+str(number_of_blocks)+'_filtered', 'rb') as fp:
    df = pickle.load(fp)
print(len(df['Sender Address'].tolist()))
G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
temp = list(G.degree())
degree_list = []
for element in temp:
    degree_list.append(element[1])

# print(degree_list)

plt.hist(degree_list, bins = 21, range=(0,20))
plt.show()