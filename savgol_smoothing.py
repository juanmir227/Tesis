import pickle
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os
import numpy as np
from progress.bar import Bar
from scipy.signal import savgol_filter

with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_filtered_percentage_of_total_transactions",'rb') as f:
    list = pickle.load(f)


transaction_type_list = ['pay', 'axfer', 'appl','acfg','keyreg', 'afrz']


with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_data_lists', 'rb') as fp:
    created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_filtered_percentage_of_total_transactions', 'rb') as f:
    filtered_percentage_of_total_transactions = pickle.load(f)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_dates', 'rb') as file:
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

print(smoothing_lists[1])
print(data_lists[1])
for list in smoothing_lists:
    for i in range(len(data_lists)):
        if list is data_lists[i]:
            data_lists[i] = savgol_filter(list,30,7)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo3_lists\periodo3_data_lists_smoothed', 'wb') as fp:
    pickle.dump(data_lists,fp)

