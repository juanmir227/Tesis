import os
import pickle
from dotenv import load_dotenv
load_dotenv()
from functions.txns_dataframe import make_df
import pandas as pd
from datetime import datetime

initial_block_number = 9000000
number_of_blocks = 500

addresses = []
applications = []

for additive in range(0,15000000,1000000):

        df = pd.read_pickle(os.environ["SAVE_DF_PATH"] + "_" + str(initial_block_number + additive) + "_" + str(number_of_blocks) + "_filtered")
        only_apps = df[df['Transaction Type'] == 'appl']
        receivers = only_apps['Receiver Address'].tolist() 
        for reciever in receivers:
                if type(reciever) == str:
                        addresses.append(reciever)
                else:
                        applications.append(reciever)
        print(receivers)


print(f"Amount of non-creation transactions: {len(applications)}")
print(f"Amount of app creation transactions: {len(addresses)}")

non_creation_percentage = len(addresses)/(len(addresses) + len(applications))
creation_percentage = 1 - non_creation_percentage

print(f"Percentage of noncreation_percetage: {creation_percentage}")
print(f"Percentage of creation transactions: {non_creation_percentage}")