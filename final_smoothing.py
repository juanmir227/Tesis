import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd

def smooth(given_list, window):
    final = []
    for i,element in enumerate(given_list):
        if 0+window<given_list.index(element)<len(given_list)-window:
        # if window<given_list.index(element)<len(given_list)-window:
            element = sum(given_list[i-window-1:i+window])/len(given_list[i-window-1:i+window])
            final.append(round(element,2))
            print(round(element,2))
        else:
            pass
    return final

def smooth_type(given_list):
    for type in transaction_type_list:
        for i,element in enumerate(given_list):
            try:
                if element == given_list[0] or element == given_list[-1] or element == given_list[1] or element == given_list[-2]:
                    pass
                else:
                    given_list[i][type] = (given_list[i-2][type] + given_list[i-1][type] + element[type] + given_list[i+1][type] + given_list[i+2][type])/5
                    given_list[i][type] = round(given_list[i][type],2)
            except:
                pass


transaction_type_list = ['pay', 'axfer', 'appl','acfg','keyreg', 'afrz']


with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_lists_csv\periodo3_data_lists', 'rb') as fp:
    created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_lists_csv\periodo3_filtered_total_transaction_percentage', 'rb') as f:
    filtered_total_transaction_percentage = pickle.load(f)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_lists_csv\periodo3_dates', 'rb') as file:
    chunk_dates = pickle.load(file)

data_lists = [created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,
mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,
only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,
sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,
most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,
percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,
percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,
more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions,total_activity]

smoothing_lists = [created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender, 
                   mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,
                   only_receiver_accounts, percent_of_senders_only_senders,percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,
                    sender_average_transacted_accounts, receiver_average_transacted_accounts, sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,
                    transactions_one_algo, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo, more_than_one_algo_percentage,
                    mean_amount_of_algo_sent]




smoothing_type_lists = [involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, percentage_of_all_transactions_per_type]


# print(mean_amount_of_algo_sent)

# print(smooth(mean_amount_of_algo_sent,2))
# for given_list in smoothing_lists:

#     smooth(given_list)

# for given_list in smoothing_type_lists:

#     smooth_type(given_list)

# for element in transaction_type_percentages_of_total_transactions:

#     smooth(element)
# window = 5
# i = 7
# given_list = mean_amount_of_algo_sent
# print(sum(given_list[i-window-1:i+window])/len(given_list[i-window-1:i+window]))

print(mean_amount_of_algo_sent)
maoas = smooth(mean_amount_of_algo_sent,7)

print(maoas)