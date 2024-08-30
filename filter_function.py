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

blacklisted_accounts = ['SN3FSQVLLNWSI3I7QCMQEQN6L42OW5RP4FNM6BNOZHBRKHODSQYU4MUPKE', '5K6J3Z54656IR7YY65WNJT54UW6RBZZYL5LWQUTG4RWOTRTRBE2MR2AODQ',
'ZW3ISEHZUHPO7OZGMKLKIIMKVICOUDRCERI454I3DB2BH52HGLSO67W754','YKYAFMDGGYCUIAGSBOSEZYJAZG6WRHHTZI3CMBEDLW7SKKRRTGGQAGYCHA',
"5TSQNIL54GB545B3WLC6OVH653SHAELMHU6MSVNGTUNMOEHAMWG7EC3AA4","ROMSEERGZW4YKYGP4UGA2LSVJXZXQTK77XB55RVHMCM6XF45RAYNKODZ7M",
"62FMJKL2BDSYWSF3RYYZHAXA5HICQ57NFZCWWNM4ZJBYSVXG6L5VC647UY","M645L4V46IHAZI23JOGCSJ4WYOVCQJ4MXC2NHRGTEUN7GIESD6GNYA7REM",
"RDKDV7CVOXLHO2OXBJCSLFDNJBTENDVT3LMFRYTZS7EKIAMAXZKBL42KHA","4VL56VPNTSFOPTI5RH45GVMM365LIUB2D3V6OBEV5HS2MKB5RTSYDD6CCY",
"SVA6PT5ILWRH2XL5R55GBDQ5FS2SGXY66RGPE5CO5ZYZMFI5ULZYBBPJ5E","GULDQIEZ2CUPBSHKXRWUW7X3LCYL44AI5GGSHHOQDGKJAZ2OANZJ43S72U",
"XFYAYSEGQIY2J3DCGGXCPXY5FGHSVKM3V4WCNYCLKDLHB7RYDBU233QB5M",389002307,"X2W76H7A57BNGV6UQNMYQHCFOK4BI4DE6AG7V7BIGIYSNGCPBO44JXRMHA",
"DETSDVFUKEFAZL5AXRB3LPFMN7IK6ONNWVCJTLSY4GNZGSBXADJJ3EQGQQ","QLLLYBITHLFUX3BWLPAXD23SBMLUYHGCG6NOPOBWY7KQHBLHLC3JC7LVBA",
"FC6NUXO5DQSOOKFN4FTDP5XWFFJZTPDQ33FQDDF6AJ4GATT6TVBQMUMSSY","UWA2JD2INPVCID3QEX5FHS7HRAS5W7F3K3BXU5OYAYYDOUPSKSZYGXBUZ4",
"ZSV653RYTKEOMS5PJA3KVGCKSEXP5XD36PTGCJFLOFC652I5QZHGD3FBMQ","37QYBAUJDHJ6GPO4X6VML733ZIYFBZTQXFR7HTFGQOQVOVJE7L4ALYUZ5A",
"ZONEGRWBV3Q7JA6RHAN4EMAX6ICIVZX2C6U65DCNHYLIL4PUBB7O6DOSBI","NW6PSNWSPPWOKDQ7WGATTCNM4B7KLNHNXIY44R7DT6TBRC3QHF46C4FHRA",
"GLGBONZR73KNY6YQFYLCC2ITBWIUCMKMUBPI3J6GXQ4EAGRH5ZWJMH5KAQ","33TEPJ2V7LVEUF5UJ4XZKTNJDZ2THE67TH7BRJKQZH6ZPKLPWKE4DWGMAI",
"BVMEUTF37WNEQ6GYCZISRFHGLEMOKT5OCPPTTJXVED6JBSXKF6YJJRZRI4","UU7C3K2IP5DCGL5Q4SZHCZNL32DRBYTZY7V4WPU7RPWAAVXIIX6GNUSFQA",
"WQF4QB4RVSVTUPQRCF3T5CKQ4ZAG7XJ7WIA523OKRMXOKEBPEVPYRDN7VY","YEX2F7BU6PTIXEDQKB4S5VVGDCBOGPXAACQXSUKQTFTFDID4LOIRR3Q4ZM",
"FAUC7F2DF3UGQFX2QIR5FI5PFKPF6BPVIOSN2X47IKRLO6AMEVA6FFOGUQ","XMP53MUN4CPNUAPKEWKC2SJ65YHSPJPHABZBJP7TYLXJ6FFYBXIEGPJWIM",
"YEPEZGWNLTRITFBX3UZWOTFMT2LY3OHDKQC4Q6OK5T7T7MHO4X2JAO6V64","ZQW3ZVL5PB5P76NWXFJSI34S3V5SF6F2YF7E4S7I7L7WNAR7C2MTXNHN2I",
"YVG5656ZA7M4QQUZHUJWV4VO6CM432LDUTKKYL6CJJLQVLSRHALMZ4MFKQ","PJLPUBJMHDYKL2EYGICXWSASANWTTQA7DBQTH3UJQTQDIA7LEV6M6BHQVY",
"L3E67NGZMM5G7PXFV377HJZBJJ335F32GTA77JAIOBXAXPZF2ZRNEPH2MA","QXE3ITCREUKZXA57VZ3KOHFSTENVWE7FLLHZ5ITJ2BVTCUJ3YTUZRL2TNM",
"QKPJSRYAQVM2QQPYNUANGLMAHKSMKQNH4DUAJM6ZPRDG6MAY6KZLWFSIYE",]


def filtering(initial_number, number_of_blocks):
    dir = r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_data_frame'
    file = "df__" + str(initial_number)+"_"+str(number_of_blocks)
    path = os.path.join(dir,file)
    df = pd.read_pickle(path)
    total_transactions = len(df['Sender Address'].tolist())
    for account in blacklisted_accounts:
        df = df[df['Sender Address'] != account]
        df = df[df['Receiver Address'] != account]
        final_filtered = df[df['Sender Address'] != df['Receiver Address']]
    total_transactions_filtered = len(final_filtered['Sender Address'].tolist())
    fraction = total_transactions_filtered/total_transactions
    percentage = round(100*fraction,2)
    dir = r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_data_frame'
    file = 'df_' + str(initial_number) + "_" + str(number_of_blocks)+ '_filtered'
    filtered_path = os.path.join(dir,file)
    final_filtered.to_pickle(filtered_path)
    G = nx.from_pandas_edgelist(final_filtered,'Sender Address', 'Receiver Address')
    dir = r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_gephi'
    file = 'gephi_' + str(initial_number) + "_" + str(number_of_blocks) + "_filtered.gexf"
    write_path = os.path.join(dir,file)
    nx.write_gexf(G,write_path)

    list_path = os.path.join(r'D:\Archivos de Programa\Carpetas\Coding\Algorand\Tesis\Tesis\periodo1_lists_csv', 'filtered_total_transaction_percentage')
    if initial_number == 11000000:
        percentages = []
        percentages.append(percentage)
        with open(list_path, 'wb') as fp:
            pickle.dump(percentages, fp)
    else:
        with open(list_path, 'rb') as f:
            percentages = pickle.load(f)
        percentages.append(percentage)
        with open(list_path, 'wb') as fp:
            pickle.dump(percentages, fp)
    return final_filtered, percentages
    # return final_filtered

init_block = 11000000
final_block = 14600000
increment = 50000
number_of_blocks = 500
initial_numbers = list(range(init_block ,final_block + increment, increment))

for init_number in initial_numbers:
    df,percentages = filtering(init_number,number_of_blocks)
print(percentages)



#periodo 1: 11000000-14600000
#periodo 2: 14800000-17400000
#periodo 3: 17600000-23000000
#Hice incrementos de 50000 bloques y saque 500 bloques

##NOTAS

## 15400000 Hay un spammer un poquito conectado pero quedan muy pocos datos en la red si lo borro
## 15500000 Sigue el mismo ahora se conecta con otro mini spammer
## 15850000 Hay otro conectado poquito
## 17950000 Aparece un faucet que tambien distribuye FAUC7F2DF3UGQFX2QIR5FI5PFKPF6BPVIOSN2X47IKRLO6AMEVA6FFOGUQ

#Finalizado el filtrado termine con 38 cuentas filtradas

#IDEA: Que pasa si saco a los agentes mas centrales de la red? Se cae el nivel de actividad por falta de conexion
# entre agentes? Cuan vulnerable es la red a la desaparicion de algun agente importante? Ante algun hackeo o algo
# asi? Como se puede cuantificar eso?

# NOTAS: A partir de 20600000 aparece una cuenta que distribuye pero que tambien esta MUY conectada con todo lo
# demas. Que se puede hacer en ese caso? Se borra o no?

#NOTAS: Parece haber un crecimiento en el avg path length pero una disminucion en las transacciones totales
# implica que aunque haya menos transacciones la red esta mas conectada que antes