import os
import pickle
from dotenv import load_dotenv
load_dotenv()
import pandas as pd

initial_block_number = 9000000
number_of_blocks = 500

creation = []
reconfiguration = []

for additive in range(0,15000000,1000000):

        df = pd.read_pickle(os.environ["SAVE_DF_PATH"] + "_" + str(initial_block_number + additive) + "_" + str(number_of_blocks) + "_filtered")
        only_assets = df[df['Transaction Type'] == 'acfg']
        ids = only_assets['Asset Id'].tolist()
        for id in ids:
                if type(id) == str:
                        creation.append(id)
                else:
                        reconfiguration.append(id)


print(f"Amount of non-creation transactions: {len(reconfiguration)}")
print(f"Amount of asset creation transactions: {len(creation)}")

creation_percentage = len(reconfiguration)/(len(creation) + len(reconfiguration))
non_creation_percentage = 1 - creation_percentage

print(f"Percentage of noncreation_percetage: {creation_percentage}")
print(f"Percentage of creation transactions: {non_creation_percentage}")