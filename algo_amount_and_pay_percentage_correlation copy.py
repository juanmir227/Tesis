import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sb
from datetime import time
import os
import pandas as pd


with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_data_frame\df_11000000_500_filtered', 'rb') as file:
    data = pd.read_pickle(file)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_lists_csv\periodo1_data_lists', 'rb') as fp:
    created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)
periodo1_pay_percentage = transaction_type_percentages_of_total_transactions[0]
periodo1_mean_amount = mean_amount_of_algo_sent

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo2_lists_csv\periodo2_data_lists', 'rb') as fp:
    created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)

periodo2_pay_percentage = transaction_type_percentages_of_total_transactions[0]
periodo2_mean_amount = mean_amount_of_algo_sent
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_lists_csv\periodo3_data_lists', 'rb') as fp:
    created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)

periodo3_pay_percentage = transaction_type_percentages_of_total_transactions[0]
periodo3_mean_amount = mean_amount_of_algo_sent

mean_amount = periodo1_mean_amount+periodo2_mean_amount+periodo3_mean_amount
pay_percentage = periodo1_pay_percentage+periodo2_pay_percentage+periodo3_pay_percentage

print(np.corrcoef(mean_amount,pay_percentage)[0,1])

#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques

initial_block1 = 11000000
final_block1 = 14600000
initial_block2 = 14800000
final_block2=17400000
initial_block3 =17600000
final_block3 = 23000000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block1, final_block1 + increment, increment))+list(range(initial_block2, final_block2 + increment, increment)) +list(range(initial_block3, final_block3 + increment, increment))

chunk_dates = []
chunk_mean_prices = []
total_amounts = []
for init_number in initial_number:
    if init_number <= 14600000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_data_frame\df_"
    elif 14800000<=init_number<=17400000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo2_data_frame\df_"
    elif init_number >= 17600000:
        path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_data_frame\df_"
    data = pd.read_pickle(path+str(init_number)+"_"+str(number_of_blocks)+"_filtered")
    pay_data = data[data['Transaction Type'] == 'pay']
    amount = pay_data['Transaction Amount'].tolist()
    amount = [float(x) for x in amount if x != 'NA']
    multiplier = 0.000001
    algo_amount = []
    algo_amount = [round(element*multiplier,2) for element in amount]
    total_amount = round(sum(algo_amount,2))
    # aca ya resolvi cuanto algo se envio en este chunk, ahora defino una fecha
    dates = data["Transaction Date"].tolist()
    new_dates = [date.date() for date in dates]
    chunk_date = new_dates[0]
    chunk_dates.append(chunk_date)
    total_amounts.append(total_amount)

# print(total_amounts)
    
plt.plot(total_amounts)
# plt.ylim(0,10000000)
plt.show()