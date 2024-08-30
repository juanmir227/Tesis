from enum import unique
import pickle
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import networkx as nx
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from progress.bar import Bar
from new_check_activity import check_activity


#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques



initial_block = 11000000
final_block = 14600000
number_of_blocks = 500
increment = 50000
initial_number = list(range(initial_block, final_block + increment, increment))

total_transaction_amount = []
total_sender_number = []
total_receiver_number = []
total_active_accounts = []
mean_transaction_amount_per_sender = []
mean_transaction_amount_per_receiver = []
mean_amount_of_unique_receiver_for_sender = []
mean_amount_of_unique_sender_for_receiver = []
only_sender_accounts = []
only_receiver_accounts = []
percent_of_senders_only_senders = []
percent_of_receivers_only_receivers = []
percent_of_accounts_only_senders = []
percent_of_accounts_only_receivers = []
sender_average_transacted_accounts = []
receiver_average_transacted_accounts = []
sender_average_transacted_with_same_accounts = []
receiver_average_transacted_with_same_accounts = []
most_frequent_ids = []
percentage_of_total_transactions_per_asset = []
unique_senders_per_asset = []
unique_receivers_per_asset = []
unique_accounts_per_asset = []
percentage_of_total_accounts_per_asset = []
transactions_one_algo = []
involved_accounts_per_type = []
involved_senders_per_type = []
involved_receivers_per_type = []
percentage_of_total_accounts_per_type = []
transaction_amount_in_microalgo = []
closing_transactions_count = []
more_than_one_algo = []
more_than_one_algo_percentage = []
mean_amount_of_algo_sent = []
created_apps = []
percentage_of_all_transactions_per_type = []
transaction_type_list = ['pay', 'axfer', 'appl','acfg','keyreg', 'afrz']
pay_percentage = []
axfer_percentage = []
appl_percentage = []
acfg_percentage = []
keyreg_percentage = []
afrz_percentage = []
percentages_per_type = []


#checkeo actividad
path = r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo1_dataframe\periodo1_"
total_activity = check_activity(initial_block,final_block, increment, number_of_blocks, path)

#ahora si corro todo el programa
bar = Bar('Processing...', max=len(initial_number))

for init_number in initial_number:
    df = pd.read_pickle(path+str(init_number)+"_"+str(number_of_blocks))

    sender = df['Sender Address'].tolist()
    receiver = df['Receiver Address'].tolist()
    total = sender + receiver

    # new_df = df.groupby(['Receiver Address'])['Sender Address'].count().reset_index(
    #   name='Count').sort_values(['Count'], ascending=False)

    # Voy a realizar estadistica descriptiva sobre todas las transacciones que tengo

    total_transactions = len(sender)
    # Numero de transacciones totales en 500 bloques
    #print(f'Total number of transactions: {total_transactions}')
    # Numero de Senders
    sender_unique = list(set(sender))
    #print(f'Total number of senders: {len(sender_unique)}')
    # Numero de Receivers
    receiver_unique = list(set(receiver))
    #print(f'Total number of receivers: {len(receiver_unique)}')
    # Numero total de cuentas activas en este chunk
    total_accounts = list(set(total))
    #print(f'Total active accounts in this chunk: {len(total_accounts)}')

    # Numero de transacciones promedio por sender
    txn_per_sender= []
    unique_connections_per_sender= []
    for sender in sender_unique:
        txns = df[df['Sender Address'] == sender]
        received_txns_addresses = txns['Receiver Address'].tolist()
        unique_received_txns_addresses = list(set(received_txns_addresses))
        txn_number = len(received_txns_addresses)
        unique_txn_number = len(unique_received_txns_addresses)

        txn_per_sender.append(txn_number)
        unique_connections_per_sender.append(unique_txn_number)
    txn_per_sender = np.array(txn_per_sender)
    unique_connections_per_sender = np.array(unique_connections_per_sender)
    mean_txn_per_sender = np.mean(txn_per_sender)
    mean_unique_connections_per_sender = np.mean(unique_connections_per_sender)

    #print(f'Mean amount of transactions per sender: {round(mean_txn_per_sender,2)}')
    #print(f'Mean amount of unique receiver accounts for a given sender: {round(mean_unique_connections_per_sender,2)}')



    # Numero de transacciones promedio por receiver
    txn_per_receiver= []
    unique_connections_per_receiver= []
    for receiver in receiver_unique:
        txns = df[df['Receiver Address'] == receiver]
        sent_txns_addresses = txns['Sender Address'].tolist()
        unique_sent_txns_addresses = list(set(sent_txns_addresses))
        txn_number = len(sent_txns_addresses)
        unique_txn_number = len(unique_sent_txns_addresses)

        txn_per_receiver.append(txn_number)
        unique_connections_per_receiver.append(unique_txn_number)

    txn_per_receiver = np.array(txn_per_receiver)
    unique_connections_per_receiver = np.array(unique_connections_per_receiver)
    mean_txn_per_receiver = np.mean(txn_per_receiver)
    mean_unique_connections_per_receiver = np.mean(unique_connections_per_receiver)

    #print(f'Mean amount of transactions per receiver: {round(mean_txn_per_receiver,2)}')
    #print(f'Mean amount of unique sender accounts for a given receiver: {round(mean_unique_connections_per_receiver,2)}')

    # De el total de cuentas, cuantas son solo senders, cuantas solo receivers, cuantas ambas?
    ### Para senders
    only_sender = []
    only_receiver = []

    for sender in sender_unique:
        receiving_sender = df[df['Receiver Address'] == sender]
        receiving_sender = receiving_sender['Receiver Address'].tolist()
        if len(receiving_sender) == 0:
            only_sender.append(sender)

    #print(f'Number of only sender accounts: {len(only_sender)}')

    sender_proportion = len(only_sender)/len(sender_unique)
    sender_total_proportion = len(only_sender)/len(total_accounts)
    #print(f'{round(sender_proportion*100,2)}% percent of senders are only senders')
    #print(f'{round(sender_total_proportion*100,2)}% percent of accounts are only senders')

    ### Para receivers


    for receiver in receiver_unique:
        sending_receiver = df[df["Sender Address"] == receiver]
        sending_receiver = sending_receiver['Sender Address'].tolist()
        if len(sending_receiver) == 0:
            only_receiver.append(receiver)

    #print(f'Number of only receiver accounts: {len(only_receiver)}')
    receiver_proportion = len(only_receiver)/len(receiver_unique)
    receiver_total_proportion = len(only_receiver) / len(total_accounts)
    #print(f'{round(receiver_proportion*100,2)}% percent of receivers are only receivers')
    #print(f'{round(receiver_total_proportion*100,2)}% percent of accounts are only receivers')

    # Numero de cuentas unicas con las que se comunica un sender

    # Numero de cuentas unicas con las que se comunica un receiver

    # De las cuentas con las que interactua, cuantas veces se comunica en promedio con la misma cuenta?


    sender_comms_count = [] # aca guardo con cuantas cuentas se comunica un sender
    receiver_comms_count = [] #aca guardo con cuantas cuentas se comunica un receiver
    mean_accounts_senders = [] # aca guardo cuantas veces se comunica con un mismo receiver cada sender
    mean_accounts_receivers = [] #aca guardo cuantas veces se comunica con un mismo sender cada receiver

    for sender in sender_unique:
        sender_filtered = df[df['Sender Address'] == sender]
        receivers = sender_filtered['Receiver Address'].tolist()
        sender_comms_count.append(len(list(set(receivers))))
        for receiver in list(set(receivers)):
            counter = []
            counter.append(receivers.count(receiver))
        counter = np.array(counter)
        mean_accounts_senders.append(np.mean(counter))

    mean_accounts_senders = np.array(mean_accounts_senders)
    real_mean_senders = np.mean(mean_accounts_senders)


    for receiver in receiver_unique:
        receiver_filtered = df[df['Receiver Address'] == receiver]
        senders = receiver_filtered['Sender Address'].tolist()
        receiver_comms_count.append(len(list(set(senders))))
        for sender in list(set(senders)):
            counter = []
            counter.append(senders.count(sender))
        counter = np.array(counter)
        mean_accounts_receivers.append(np.mean(counter))

    mean_accounts_receivers = np.array(mean_accounts_receivers)
    real_mean_receivers = np.mean(mean_accounts_receivers)
    mean_sender_comms_count = np.mean(np.array(sender_comms_count))
    mean_receivers_comms_count = np.mean(np.array(receiver_comms_count))

    #print(f'A given sender in average transacts with {round(mean_sender_comms_count,2)} accounts')
    #print(f'A given receiver in average transacts with {round(mean_receivers_comms_count,2)} accounts')
    #print(f'A given sender in average transacts with the same account {round(real_mean_senders,2)} times')
    #print(f'A given receiver in average transacts with the same account {round(real_mean_receivers,2)} times')
    # Cual es el asset id que mas se repite? Cuanto porcentaje de las transacciones involucran a cada asset?

    asset_id = df['Asset Id'].tolist()

    ids = []
    for i in range(len(asset_id)):

        id = asset_id[i]
        if id != 'NA':
            ids.append(id)

    for id in ids:
        id = str(id)
    ids = pd.DataFrame(data= {"Asset Id": ids})
    asset_txn = len(ids['Asset Id'].tolist())


    counter = ids.groupby(['Asset Id'])['Asset Id'].count().reset_index(
    name='Count').sort_values(['Count'], ascending=False)
    counter = counter[:10]

    Ids = counter['Asset Id'].tolist()
    counts = counter['Count'].tolist()

    ticks = range(len(Ids))

    plt.bar(ticks, counts, align = 'center')
    plt.xticks(ticks, Ids, rotation = 'vertical')

    asset_total_proportion = np.array(counts)/asset_txn
    #print(f'Top 10 most frequently used Assets Ids :{Ids}')
    #print(f'Percenteage of total transactions involving each asset: {asset_total_proportion*100}')

    unique_accounts = []
    unique_sending = []
    unique_receiving = []

    # Cuantas cuentas unicas usan ese mismo asset? Cual es el promedio de cuentas que usan cada asset? Histograma de eso
    for id in Ids:

        filtered = df[df['Asset Id'] == id]
        sending = filtered['Sender Address'].tolist()
        receiving = filtered['Receiver Address'].tolist()
        unique_sending.append(len(list(set(sending))))
        unique_receiving.append(len(list(set(receiving))))
        total = sending + receiving
        unique_accounts.append(len(list(set(total))))

    #print(f'Number of unique senders for each asset: {unique_sending}')
    #print(f'Number of unique receivers for each asset: {unique_receiving}')
    #print(f'Number of unique accounts for each asset: {unique_accounts}')

    account_proportion = np.array(unique_accounts)/len(total_accounts)

    #print(f'Percentage of total accounts using this given asseta for each asset: {account_proportion*100,2}')

    # Numero de transacciones que involucran un amount igual a 0, menor a 1, que tipo de asset se mandan?
    amount = df['Transaction Amount'].tolist()
    amt = []
    for i in range(len(amount)):
        txn = amount[i]
        if txn != 'NA':
            amt.append(txn)

    def condition(x):
        return x == 1

    amount_1 = sum(map(condition, amt))
    #print(f'Number of transaction which send 1 of an Asset: {amount_1}')

    one = df[df['Transaction Amount'] == 1]
    what_asset = one['Asset Id'].tolist()

    #print(what_asset.count(127746157))
    #print(list(set(what_asset)))

    #Por lo que se ve ademas de la 127746786 moneda esta del coso de ajedrez la mayoria de las demas son NFTs o assets unicos.

    # Por cada tipo de transaccion, cuantas cuentas estan involucradas? Cuanto porcentaje del total de las transacciones en este chunk?

    # Cual es el promedio de senders y receivers por cada tipo de transaccion?


    types = df['Transaction Type']
    txn_type = list(set(types))
    #print(f'Tipos de transaccion: {txn_type} \n')

    senders_per_type = {}
    receivers_per_type = {}
    accounts_per_type = {}
    percentage_for_type = {}
    total_accounts_involved = []
    for txn_types in txn_type:
        all_transactions = len(df['Sender Address'].tolist())
        df_with_type = df[df['Transaction Type'] == txn_types]
        amount_of_transactions_per_type = len(df_with_type['Receiver Address'].tolist())
        percentage_per_type = amount_of_transactions_per_type/all_transactions
        senders_involved = df_with_type['Sender Address'].tolist()
        receivers_involved = df_with_type['Receiver Address'].tolist()
        total_involved = senders_involved + receivers_involved
        percentage_for_type[txn_types] = percentage_per_type
        accounts_per_type[txn_types] = len(list(set(total_involved)))
        senders_per_type[txn_types] = len(list(set(senders_involved)))
        receivers_per_type[txn_types] = len(list(set(receivers_involved)))

    #print(f'Total involved accounts per transaction type: {accounts_per_type} \n')
    #print(f'Total senders involved per transaction type : {senders_per_type}\n')
    #print(f'Total receivers involved per transaction type: {receivers_per_type} \n')
    percent_accounts_per_type = {}
    for key in accounts_per_type:
        number = accounts_per_type[key]
        percent_accounts_per_type[key] = round(number/len(total_accounts)*100,2)
    #print(f'Percentage of all accounts involved for each transaction type {percent_accounts_per_type}')

    ### Dentro del tipo pay que pasa? la gente se manda algos?

    payables = df[df['Transaction Type'] == 'pay']

    payable_senders = payables['Sender Address']
    payable_receivers = payables['Receiver Address']
    list(set(payable_senders))

    payable_amount = payables['Transaction Amount'].tolist()
    transaction_amount = payable_amount
    #print(f'Transaction amount: {len(transaction_amount)}')

    closing_transactions = payable_amount.count('NA')

    #print(f'Closing transactions: {closing_transactions}')
    placeholder = []

    for i in range(len(payable_amount)):
        amounts = payable_amount[i]
        if amounts != 'NA':
            placeholder.append(amounts)

    payable_amount = placeholder
    # print(len(payable_amount))

    payable_amount = np.array(payable_amount)

    mean_amount = np.mean(payable_amount)
    mean_amount = mean_amount/1000000
    mean_amount = round(mean_amount, 2)

    # plt.hist(payable_amount, bins = 1000)
    # plt.xlim([0,0.8*10**10])  


    amount_counter = payables.groupby(['Transaction Amount'])['Transaction Amount'].count().reset_index(
    name='Count').sort_values(['Count'], ascending=False)

    #amount_counter = amount_counter.drop([1056])
    # print(amount_counter)
    # Checkeo cuantas transacciones mandan mas de un algo

    more_than_one = payable_amount[payable_amount >= 1000000]
    # print(more_than_one)
    #print(f'Number of transactions sending more than 1 Algo: {len(more_than_one)}') #831 de las transacciones son envios mayores que 1 algo
    #print(f'On average each with each transaction {mean_amount} Algos are being sent')

    frac_more_than_one = len(payable_amount) / len(df['Sender Address'].tolist())
    percent_more_than_one = frac_more_than_one*100
    percent_more_than_one = round(percent_more_than_one,2)
    # print(frac_more_than_one*100)

    #print(f'{percent_more_than_one} % of total transactions are type pay and send more than 1 Algo')
    apps = df[df["Transaction Type"] == 'appl']
    app_receivers = apps['Receiver Address'].tolist()
    apps_created = app_receivers.count('NA')

    ## TODAVIA FALTA AGREGAR TODO ESTO DE ABAJO. ANTES DE CORRER Y JUNTAR TODA LA DATA

    type_percentages = []
    transaction_type = df['Transaction Type'].tolist()
    for type in transaction_type_list:
        
        temp = transaction_type.count(type)
        percentage = temp/len(transaction_type)*100
        type_percentages.append(percentage)
    percentages_per_type.append(type_percentages)

    created_apps.append(apps_created)
    total_transaction_amount.append(total_transactions)
    total_sender_number.append(len(sender_unique))
    total_receiver_number.append(len(receiver_unique))
    total_active_accounts.append(len(total_accounts))
    mean_transaction_amount_per_sender.append(round(mean_txn_per_sender,2))
    mean_transaction_amount_per_receiver.append(round(mean_txn_per_receiver,2))
    mean_amount_of_unique_receiver_for_sender.append(round(mean_unique_connections_per_sender,2))
    mean_amount_of_unique_sender_for_receiver.append(round(mean_unique_connections_per_receiver,2))
    only_sender_accounts.append(len(only_sender))
    only_receiver_accounts.append(len(only_receiver))
    percent_of_senders_only_senders.append(round(sender_proportion*100,2))
    percent_of_receivers_only_receivers.append(round(receiver_proportion*100,2))
    percent_of_accounts_only_senders.append(round(sender_total_proportion*100,2))
    percent_of_accounts_only_receivers.append(round(receiver_total_proportion*100,2))
    sender_average_transacted_accounts.append(round(mean_sender_comms_count,2))
    receiver_average_transacted_accounts.append(round(mean_receivers_comms_count,2))
    sender_average_transacted_with_same_accounts.append(round(real_mean_senders,2))
    receiver_average_transacted_with_same_accounts.append(round(real_mean_receivers,2))
    most_frequent_ids.append(Ids)
    percentage_of_total_transactions_per_asset.append(asset_total_proportion*100)
    unique_senders_per_asset.append(unique_sending)
    unique_receivers_per_asset.append(unique_receiving)
    unique_accounts_per_asset.append(unique_accounts)
    percentage_of_total_accounts_per_asset.append(np.round_(account_proportion*100,decimals = 2))
    transactions_one_algo.append(amount_1)
    involved_accounts_per_type.append(accounts_per_type)
    involved_senders_per_type.append(senders_per_type)
    involved_receivers_per_type.append(receivers_per_type)
    percentage_of_total_accounts_per_type.append(percent_accounts_per_type)
    transaction_amount_in_microalgo.append(len(transaction_amount))
    closing_transactions_count.append(closing_transactions)    
    more_than_one_algo.append(len(more_than_one))
    more_than_one_algo_percentage.append(percent_more_than_one)
    mean_amount_of_algo_sent.append(mean_amount)
    percentage_of_all_transactions_per_type.append(percentage_for_type)
    bar.next()
bar.finish()

for element in percentages_per_type:

        pay_percentage.append(element[0])
        axfer_percentage.append(element[1])
        appl_percentage.append(element[2])
        acfg_percentage.append(element[3])
        keyreg_percentage.append(element[4])
        afrz_percentage.append(element[5])


transaction_type_percentages_of_total_transactions = [pay_percentage, axfer_percentage, appl_percentage, acfg_percentage, keyreg_percentage, afrz_percentage]

data_lists = [created_apps, total_transaction_amount, total_sender_number, total_receiver_number, total_active_accounts, mean_transaction_amount_per_sender,
mean_transaction_amount_per_receiver, mean_amount_of_unique_receiver_for_sender, mean_amount_of_unique_sender_for_receiver, only_sender_accounts,
only_receiver_accounts, percent_of_senders_only_senders, percent_of_receivers_only_receivers, percent_of_accounts_only_senders, percent_of_accounts_only_receivers,
sender_average_transacted_accounts, receiver_average_transacted_accounts,sender_average_transacted_with_same_accounts, receiver_average_transacted_with_same_accounts,
most_frequent_ids, percentage_of_total_transactions_per_asset, unique_senders_per_asset, unique_receivers_per_asset, unique_accounts_per_asset,
percentage_of_total_accounts_per_asset, transactions_one_algo, involved_accounts_per_type, involved_senders_per_type, involved_receivers_per_type,
percentage_of_total_accounts_per_type, transaction_amount_in_microalgo, closing_transactions_count, more_than_one_algo,
more_than_one_algo_percentage, mean_amount_of_algo_sent, percentage_of_all_transactions_per_type, transaction_type_percentages_of_total_transactions,total_activity]

with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\periodo1_lists\periodo1_data_lists', 'wb') as fp:
    pickle.dump(data_lists, fp)
