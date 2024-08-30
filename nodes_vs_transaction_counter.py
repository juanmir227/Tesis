import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sb
import pandas as pd


with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\gen_list\number_of_nodes', 'rb') as f:
    numberOfNodes = pickle.load(f)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo1_lists\periodo1_dates', 'rb') as file:
    chunk_dates1 = pickle.load(file)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo2_lists\periodo2_dates', 'rb') as file:
    chunk_dates2 = pickle.load(file)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_lists\periodo3_dates', 'rb') as file:
    chunk_dates3 = pickle.load(file)

chunk_dates = chunk_dates1 + chunk_dates2 + chunk_dates3

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo1_lists\periodo1_data_lists_smoothed', 'rb') as fp:
    created_apps, total_transaction_amount1, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo2_lists\periodo2_data_lists_smoothed', 'rb') as fp:
    created_apps, total_transaction_amount2, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Data sets 4 y 5\periodo3_lists\periodo3_data_lists_smoothed', 'rb') as fp:
    created_apps, total_transaction_amount3, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions, total_activity = pickle.load(fp)

total_transaction_amount = list(total_transaction_amount1) + list(total_transaction_amount2) + list(total_transaction_amount3)

plt.rc('figure', figsize= (15,6))
sb.set_style('darkgrid')
total_transaction_amount = savgol_filter(total_transaction_amount, 20, 7)
numberOfNodes = savgol_filter(numberOfNodes, 20, 7)
total_transaction_amount = total_transaction_amount/np.max(total_transaction_amount)
numberOfNodes = numberOfNodes/np.max(numberOfNodes)


data_data_frame = pd.DataFrame({'transaction_amount': total_transaction_amount,'nodes': numberOfNodes, 'dates': chunk_dates})
csv_data = data_data_frame.to_csv(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\Final CSVs\transaction_vs_node_count\transaction_vs_nodes.csv', index = False)



plt.plot(chunk_dates, total_transaction_amount, label = 'Número de transacciones', color = 'blue')
plt.plot(chunk_dates, numberOfNodes, label = 'Número de cuentas', color = 'red')
plt.title('Comportamiento del número de cuentas y transacciones')
plt.legend()
plt.xlabel('Fechas')
plt.ylabel('Porcentaje relativo al máximo')
plt.show()

gamma = np.corrcoef(numberOfNodes, total_transaction_amount)
print(gamma)