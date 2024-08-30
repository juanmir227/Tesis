import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import datetime



#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques


df_list = []
initial_block = 17600000
final_block = 23000000
increment = 50000
number_of_blocks = 500
dates = []
block = initial_block
initial_number = list(range(initial_block, final_block + increment, increment))

for number in initial_number:
    # print(number)
    path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_dataframe\periodo3"
    df = pd.read_pickle(path + "_" + str(number) + "_" + str(number_of_blocks))
    date = df['Transaction Date'].tolist()[0].date()
    dates.append(date)
    block = block + increment

with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_dates", 'wb') as fp:
    pickle.dump(dates, fp)