import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
import os
import pandas as pd
from scipy.signal import savgol_filter
import datetime

with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo1_lists\periodo1_dates", 'rb') as f:
    periodo1_dates = pickle.load(f)
with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo2_lists\periodo2_dates", 'rb') as f:
    periodo2_dates = pickle.load(f)
with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_dates", 'rb') as f:
    periodo3_dates = pickle.load(f)

full_dates = periodo1_dates + periodo2_dates + periodo3_dates

with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\gen_list\max_degree_list", 'rb') as f:
    max_degree = pickle.load(f)

filtered_degree = savgol_filter(max_degree, 30,7)
plt.plot(full_dates, filtered_degree, color = 'red', label = 'filtered')
plt.plot(full_dates, max_degree, color = 'pink', label = 'original')
plt.xlabel('Date')
plt.ylabel('Max Degree')
plt.show()

with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\gen_list\number_of_nodes", 'rb') as f:
    number_of_nodes = pickle.load(f)
print(number_of_nodes)
filtered_nodes = savgol_filter(number_of_nodes, 30,7)
print(filtered_nodes)
plt.plot(full_dates, filtered_nodes, color = 'red', label = 'filtered')
plt.plot(full_dates, number_of_nodes, color = 'pink', label = 'original')
plt.xlabel('Date')
plt.ylabel('Number of Nodes in the Network')
plt.show()