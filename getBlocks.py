from algosdk.v2client import algod
from dotenv import load_dotenv
load_dotenv()
from progress.bar import Bar

#round  reference 23.739.705

def GetBlockInfo(client,initial_block_number, block_number):
    blocks = []
    bar = Bar('Downloading', max=block_number)
    for i in range(block_number):
        # Retrieve block information                                                                                                                                              
        try:
            block = client.block_info(initial_block_number+i)
        except Exception as e:
            print("Failed to get algod status: {}".format(e))
        blocks.append(block)
        bar.next()
    bar.finish()
    return blocks
