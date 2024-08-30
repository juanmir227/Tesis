from algosdk.v2client import algod
import pickle
from getBlocks import GetBlockInfo
from acquireTxns import join_txns
from txnsDataframe import make_df
import pandas as pd
import networkx as nx

### Genero el cliente para descargar los bloques
algod_address = "https://mainnet-algorand.api.purestake.io/ps2"
algod_token = ""
headers = {
    "X-API-Key": "M5TMX75YWr4iICfTHJ04N5FkDdn2cfLF23M8VQRV",
}

algod_client = algod.AlgodClient(algod_token, algod_address, headers)

#defino bloque inicial y cuantos bloques quiero descargar
# 9 millones es el bloque minimo para arrancar, antes de eso no funciona.

#Esto esta para no olvidarme
#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques


initial_number = 22800000 #Acá se selecciona el bloque inicial
number_of_blocks = 20000 #Cantidad de bloques que se desea descargar
increment = 1 #Incremento de distancia entre bloques descargados
final_block = 22820000 #Bloque final al que se quiere llegar

while initial_number <= final_block:
    initial_block_number = initial_number
    print(initial_block_number)
    initial_number = initial_number + increment

    #descargo los bloques y los guardo
    blocks = GetBlockInfo(algod_client,initial_block_number,number_of_blocks)
    with open(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_block_data\blocks_"+ str(initial_block_number)+'_'+str(number_of_blocks), 'wb') as what:
        pickle.dump(blocks, what)

    #en caso de ya tener los bloques directamente los cargo
    # with open(os.environ["SAVE_BLOCK_PATH"]+"_"+str(initial_block_number)+'_'+str(number_of_blocks), 'rb') as what:
    #     blocks = pickle.load(what)

    #genero toda la lista de transacciones y las guardo
    transacciones = join_txns(blocks)
    with open(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_txn_data\txns_'+ str(initial_block_number)+"_"+str(number_of_blocks), 'wb') as fp:
        pickle.dump(transacciones, fp)

    #genero el dataframe y lo guardo
    data = make_df(transacciones)
    df = pd.DataFrame(data)
    df.to_pickle(r"D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_data_frame\df_" +"_"+str(initial_block_number)+"_"+str(number_of_blocks))

    #genero el gephi a partir del dataframe y lo guardo
    G = nx.from_pandas_edgelist(df,'Sender Address', 'Receiver Address')
    nx.write_gexf(G,r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo3_gephi\gephi_'+"_"+str(initial_block_number)+"_" + str(number_of_blocks)+'.gexf')