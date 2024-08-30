import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from progress.bar import Bar

number_of_blocks = 500
initial_block = 11000000
final_block = 14600000
increment = 50000

def check_activity(initial_block, final_block, increment, number_of_blocks,path):

    # path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_data_frame\df_"
    # df = pd.read_pickle(path+str(init_number)+"_"+str(number_of_blocks)+"_filtered")
    initial_numbers = list(range(initial_block, final_block + increment, increment))

    df_list = []
    for init_number in initial_numbers:
        path = path
        df_list.append(pd.read_pickle(path+ str(init_number) + "_" + str(number_of_blocks)))
    address = []
    df_merged = pd.concat(df_list)
    senders = df_merged['Sender Address'].tolist()
    receivers = df_merged['Receiver Address'].tolist()
    total_addresses = senders + receivers
    unique_accounts = list(set(total_addresses))
    bar = Bar('Processing...', max=len(unique_accounts))
    total_activity = []
    for account in unique_accounts:
        check = []
        for df in df_list:
            check.append(account in df['Sender Address'].tolist() or account in df['Receiver Address'].tolist())
        total_activity.append(sum(check)) 
        bar.next()
    bar.finish()

    return total_activity