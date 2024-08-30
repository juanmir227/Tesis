import pickle
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
accounts = []
degrees = []
txn_counts = []
path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\big_chunk_periodo4\big_chunk_periodo4_dataframe_filtered"
df = pd.read_pickle(path)
G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
temp = list(G.degree())
for element in temp:
    accounts.append(element[0])
    degrees.append(element[1])

for account in accounts:

    filtered = df.loc[(df['Sender Address'] == account) | (df['Receiver Address'] == account), ['Sender Address', 'Receiver Address']]
    txn_count = len(filtered['Sender Address'].tolist())
    txn_counts.append(txn_count)
    
data = [accounts, degrees, txn_counts]
with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\big_chunk_periodo4\account_degree_txncount_data_periodo4","wb") as f:
    pickle.dump(data, f)