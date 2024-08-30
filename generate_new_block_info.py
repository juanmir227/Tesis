from algosdk.v2client import algod
import pickle
from getBlocks import GetBlockInfo
from acquireTxns import join_txns
from txnsDataframe import make_df
import pandas as pd
import networkx as nx

## La idea es runearlo varias veces e iterar sobre distintos valores iniciles de bloque para tener varios de estos chunks y
## de esa manera poder comparar como van cambiando las estadisticas a medida que avanza el tiempo


### Genero el cliente para descargar los bloques
algod_address = "https://mainnet-algorand.api.purestake.io/ps2"
algod_token = ""
headers = {
    "X-API-Key": "M5TMX75YWr4iICfTHJ04N5FkDdn2cfLF23M8VQRV",
}

algod_client = algod.AlgodClient(algod_token, algod_address, headers)
###


#defino bloque inicial y cuantos bloques quiero descargar
# 9 millones es el bloque minimo para arrancar, antes de eso no funciona, se ve que estructuraban los bloques diferente
# o todavia no habia transacciones, no se la verdad.

#periodo 1: 14000000
#periodo 2: 17200000
#periodo 3: 20000000
#Hice incrementos de 50000 bloques y saque 500 bloques


initial_number = 22800000
number_of_blocks = 20000
#genero chunk de 5 bloques cada 1 millon de bloques

initial_block_number = initial_number
print(initial_block_number)
#descargo los bloques y los guardo
blocks = GetBlockInfo(algod_client,initial_block_number,number_of_blocks)
with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\big_chunk_periodo4\big_chunk_periodo4_blocks", 'wb') as what:
    pickle.dump(blocks, what)
#en caso de ya tener los bloques directamente los cargo
# with open(os.environ["SAVE_BLOCK_PATH"]+"_"+str(initial_block_number)+'_'+str(number_of_blocks), 'rb') as what:
#     blocks = pickle.load(what)
#genero toda la lista de transacciones y las guardo
transacciones = join_txns(blocks)
with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\big_chunk_periodo4\big_chunk_periodo4_transactions', 'wb') as fp:
    pickle.dump(transacciones, fp)

#genero el dataframe y lo guardo
data = make_df(transacciones)
df = pd.DataFrame(data)
df.to_pickle(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\big_chunk_periodo4\big_chunk_periodo4_dataframe")

#genero el gephi y lo guardo
G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
nx.write_gexf(G,r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\filtering_algorithm\big_chunk_periodo4\big_chunk_periodo4_gephi.gefx')