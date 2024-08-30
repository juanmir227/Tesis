import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd

percentages_per_type = []


transaction_type_list = ['pay', 'axfer', 'appl','acfg','keyreg', 'afrz']

initial_numbers = list(range(11000000,23200000,200000))
number_of_blocks = 1000
df_list = []
for init_number in initial_numbers:
    type_percentages = []
    df = pd.read_pickle(os.environ['SAVE_DF_PATH']+"_"+str(init_number)+"_"+str(number_of_blocks)+"_filtered")
    transaction_types = df["Transaction Type"].tolist()
    for type in transaction_type_list:
        temp = transaction_types.count(type)
        percentage = temp/len(transaction_types)*100
        type_percentages.append(percentage)
    percentages_per_type.append(type_percentages)    

pay_percentage = []
axfer_percentage = []
appl_percentage = []
acfg_percentage = []
keyreg_percentage = []
afrz_percentage = []


for element in percentages_per_type:

    pay_percentage.append(element[0])
    axfer_percentage.append(element[1])
    appl_percentage.append(element[2])
    acfg_percentage.append(element[3])
    keyreg_percentage.append(element[4])
    afrz_percentage.append(element[5])


transaction_type_percentages_of_total_transactions = [pay_percentage, axfer_percentage, appl_percentage, acfg_percentage, keyreg_percentage, afrz_percentage]

# with open('/home/juaneto8/Documents/Projects/Algorand/data_acquisition/new_lists_csv/' + 'transaction_type_percentages_of_total_transactions', 'wb') as fp:
#     pickle.dump(transaction_type_percentages_of_total_transactions, fp)
