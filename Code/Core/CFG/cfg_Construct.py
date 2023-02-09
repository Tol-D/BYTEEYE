from . import basic_Block as bb
from . import basic_CFG as bc
import sys
sys.path.append("..")
# from ..Detection import Reentrancy_Detect as re  #RUN main.py
from ..Contracts import contracts_Identify as ci  #RUN main.py
from ..Opcode import std_Opcodes
import csv
import json
csv.field_size_limit(500*1024*1024)
def input_contract():
    # Temporary Entrance
    print("input_contract")
    #bytecode = "608060405260aa60005534801561001557600080fd5b50610184806100256000396000f3fe608060405234801561001057600080fd5b506004361061004c5760003560e01c80634b3df200146100515780635bfd987d1461006f578063c2985578146100b1578063febb0f7e146100cf575b600080fd5b6100596100ed565b6040518082815260200191505060405180910390f35b61009b6004803603602081101561008557600080fd5b810190808035906020019092919050505061011d565b6040518082815260200191505060405180910390f35b6100b961012a565b6040518082815260200191505060405180910390f35b6100d761013c565b6040518082815260200191505060405180910390f35b6000806000815480929190600101919050555060aa600054141561011557600054905061011a565b600080fd5b90565b6000816088019050919050565b600061013761cccc61011d565b905090565b600061014961dddd61011d565b90509056fea2646970667358221220e619b234c1887f9b10b567ee21364dbf523a19001c8c47a33049907c0398563164736f6c63430006040033"
    #This sample is used for reentrancy detection
    bytecode="608060405234801561001057600080fd5b50610396806100206000396000f3fe608060405234801561001057600080fd5b506004361061004c5760003560e01c8063102ccd9214610051578063271625f71461007f578063cc06463214610089578063cd482e7714610093575b600080fd5b61007d6004803603602081101561006757600080fd5b810190808035906020019092919050505061009d565b005b61008761019d565b005b61009161023f565b005b61009b6102c7565b005b806000803373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205410156100e857600080fd5b3373ffffffffffffffffffffffffffffffffffffffff168160405180600001905060006040518083038185875af1925050503d8060008114610146576040519150601f19603f3d011682016040523d82523d6000602084013e61014b565b606091505b505050806000803373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000206000828254039250508190555050565b600073888888888888888888888888888888888888888890508073ffffffffffffffffffffffffffffffffffffffff166108fc61aaaa9081150290604051600060405180830381858888f193505050505061aaaa6000808373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000206000828254039250508190555050565b3373ffffffffffffffffffffffffffffffffffffffff166108fc6188889081150290604051600060405180830381858888f19350505050506188886000803373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008282540392505081905550565b3373ffffffffffffffffffffffffffffffffffffffff166108fc6199999081150290604051600060405180830381858888f1935050505015801561030f573d6000803e3d6000fd5b506199996000803373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000206000828254039250508190555056fea2646970667358221220df8e9a0b8321357ef3ec613ee7864e94116500d68b9bb1dba6f439f8c9dc76bc64736f6c634300060c0033"
    return bytecode
#for new SCP reentrancy dataset
def input_contract11():
    bytecode=[]
    filen=[]
    with open('Core/Input/contracts_Compiled/scpre.csv', 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            filen.append(row[1])
            bytecode.append(row[2])
    del bytecode[0]
    del filen[0]
    for index in range(len(bytecode)):
        bytecode[index]=bytecode[index]
    return bytecode,filen

#turn a str to a int
def str2int(str):
    intermediate = bytes(str, 'UTF-8')
    str_int = int.from_bytes(intermediate, 'big')
    end = 0
    if (str_int > 47 and str_int < 58):
        end = str_int - 48
    elif (str_int > 96 and str_int < 103):
        end = str_int - 87
    return end

#To deal with sample contract
#The structure is  XXfe runtime code feXX
def split_contract(bytecode):
    count = 0
    Block_index = []  # It remarks the identification of basic blocks
    Disa_bytecode = []  # It divides bytecode into different instruction
    Block_index.append(0)  # recode the first basic block
    # Split instruction from contract bytecode
    while (count < len(bytecode)):
        start = count
        # PUSH operation follows with pushed data
        if (bytecode[count] == "6" or bytecode[count] == "7"):
            temp = 0
            if (bytecode[count] == "6"):
                temp = (str2int(bytecode[count + 1]) + 1) * 2
            elif (bytecode[count] == "7"):
                temp = (str2int(bytecode[count + 1]) + 17) * 2
            count = count + temp
        count = count + 2
        Disa_bytecode.append(bytecode[start:count])
    # temporary operation: for processing the example contract
    # Need further processing!!!  TODO
    #--------------------------------
    #split runtime code from Disa_bytecode   have trouble  TODO
    Runtime_code=[]
    first_runtime = False
    for index in range(len(Disa_bytecode)):
        if(first_runtime):
            if(Disa_bytecode[index][0:2]=='fe'):
                break
            else:
                Runtime_code.append(Disa_bytecode[index])
        if(Disa_bytecode[index][0:2]=='fe'): #after 'fe', part of Runtime code starts
            first_runtime=True

    #After spliting the contract, we need to check every instruction in runtime code. TODO
    print("invalid instruction in runtime code is :")
    for index in range(len(Runtime_code)):
        hexb = '0x' + Runtime_code[index][0:2]
        if (hexb not in std_Opcodes.standard_opcodes.keys()):
            print(index)
            print(hexb)
    # END---------------------------

    return Disa_bytecode,Runtime_code

#To deal with the contracts from test_contract.csv
def split_contract2(bytecode,version):
    count = 0
    Block_index = []  # It remarks the identification of basic blocks
    Disa_bytecode = []  # It divides bytecode into different instruction
    Block_index.append(0)  # recode the first basic block
    # Split instruction from contract bytecode
    while (count < len(bytecode)):
        start = count
        # PUSH operation follows with pushed data
        if (bytecode[count] == "6" or bytecode[count] == "7"):
            temp = 0
            if (bytecode[count] == "6"):
                temp = (str2int(bytecode[count + 1]) + 1) * 2
            elif (bytecode[count] == "7"):
                temp = (str2int(bytecode[count + 1]) + 17) * 2
            count = count + temp
        count = count + 2
        Disa_bytecode.append(bytecode[start:count])
    # temporary operation: for processing the example contract

    search_fe=[]
    search_stop=[]
    print("invalid instruction in Disa_bytecode is :")
    for index in range(len(Disa_bytecode)):
        hexb = '0x' + Disa_bytecode[index][0:2]
        if (hexb not in std_Opcodes.standard_opcodes.keys()):
            print(index)
            print(hexb)
            search_fe.append([index,hexb])
        if hexb=="0x00":
            search_stop.append([index,hexb])
    # END---------------------------
    #--------------------------------

    count_fe=0
    for index in range(len(Disa_bytecode)):
        if Disa_bytecode[index]=='fe':
            count_fe=count_fe+1
    Runtime_code=[]
    # split runtime code from Disa_bytecode     TODO
    if version=='v0.5.12':
        for index in range(len(Disa_bytecode)):
            if(Disa_bytecode[index][0:2]=='fe' and version!='v0.5.12' ):
                break
            else:
                Runtime_code.append(Disa_bytecode[index])
    elif version=='v0.4.17':#for ESC_RE dataset, we have to handle the dia_bytecode part
        stopa=False
        for index in range(len(Disa_bytecode)):
            if stopa:
                Runtime_code.append(Disa_bytecode[index])
            if(Disa_bytecode[index][0:2]=='00'):
                stopa=True
            if (Disa_bytecode[index][0:2] == 'fe' and version != 'v0.5.12'and version != 'v0.4.17'):
                break
    elif version=='sbd':
        sta_less=0
        for i in range(len(search_fe)):
            if search_fe[i][1]!='0xfe':
                sta_less=search_fe[i][0]
                break
        sta_stop=0
        if sta_less!=0:
            for i in range(len(search_stop)-1,-1,-1):
                if search_stop[i][0]<sta_less:
                    sta_stop=search_stop[i][0]
                    break
            for index in range(0,sta_stop+1):
                Runtime_code.append(Disa_bytecode[index])
        else:
            for index in range(len(Disa_bytecode)):
                Runtime_code.append(Disa_bytecode[index])
        print()
    elif version=='v0.6.10':
        Runtime_code=Disa_bytecode


    #After spliting the contract, we need to check every instruction in runtime code. TODO
    print("invalid instruction in runtime code is :")
    for index in range(len(Runtime_code)):
        hexb = '0x' + Runtime_code[index][0:2]
        if (hexb not in std_Opcodes.standard_opcodes.keys()):
            print(index)
            print(hexb)
    # END---------------------------
    return Disa_bytecode,Runtime_code,search_fe


def block_identification(bytecode):
    Block_index = []  # It remarks the identification of basic blocks
    Block_index.append(0)  # recode the first basic block
    for index in range(len(bytecode)):
        #opcode 5b marks the start of a block
        if (bytecode[index][0] == "5" and bytecode[index][1] == "b"):
            if index not in Block_index:
                Block_index.append(index)
        # STOP operation
        if (bytecode[index][0] == "0" and bytecode[index][1] == "0"):
            Block_index.append(index + 1)
            # print("basic block identification")
        # JUMPI and JUMPI Operation
        elif (bytecode[index][0] == "5"):
            if (bytecode[index][1] == "6" or bytecode[index][1] == "7"):
                Block_index.append(index + 1)
                # print("basic block identification")
        # RETURN, REVERT and SELFDESTRUCT Operation
        elif (bytecode[index][0] == "f"):
            # if (bytecode[index][1] == "3" or bytecode[index][1] == "d" or bytecode[index][1] == "e" or
            #         bytecode[index][1] == "f"):
            if (bytecode[index][1] == "3" or bytecode[index][1] == "d"  or  bytecode[index][1] == "f"):
                Block_index.append(index + 1)
                # print("basic block identification")
        # Invalid Operation TODO

    # we need to define a basic class of blocks
    return Block_index


def block_build(bytecode, Block_index):
    Basic_blocks = []
    for i in range(len(Block_index)):
        if (i + 1 == len(Block_index)):
            Block = bb.BasicBlock(bytecode, i, Block_index[i], Block_index[(len(Block_index) - 1)])
        else:
            Block = bb.BasicBlock(bytecode, i, Block_index[i], Block_index[i + 1] - 1)
        if i==3:
            print("")
        Block.compute_StartStack()
        Basic_blocks.append(Block)
    return Basic_blocks


class cfg_construction:
    # Initialization Function
    def __init__(self, input_bytecode):
        self.bytecode = input_bytecode
        self.DisA_bytecode = []
        self.Runtime_bytecode=[]
        self.Block_index = []
        self.Basic_blocks = []
        #self.CFG=bc.basicCFG(NONE,NONE)

    def cfg_construct(self,num):
        print("start cfg construction")

        # the first step is to parse Solidity into EVM assemably
        print("-------------------------")
        print("input bytecode is " + self.bytecode)
        # self.DisA_bytecode,self.Runtime_bytecode = split_contract(self.bytecode) #For sample contract
        self.bytecode,version=ci.remove_metadata(self.bytecode,"sbd")#It's valid for current 0.5.12 version smart contracts
        self.DisA_bytecode, self.Runtime_bytecode,search_fe = split_contract2(self.bytecode,version)

        # the second step is basic block identification and construction
        print("-------------------------")
        self.Block_index = block_identification(self.Runtime_bytecode)
        self.Basic_blocks = block_build(self.Runtime_bytecode, self.Block_index)
        print("Block identification and construction finishes")

        # the third step is to construct the edges of control flow graph
        print("-------------------------")
        AA=bc.basicCFG(self.Basic_blocks,num)
        AA.build_CFG()
        print("CFG construction finished")
        print("-------------------------")

        #detect the precision of the whole CFG
        #SB_dataset,we stop this function
        print("start detection(depend on expert rules)")
        edge_analysis=None
        log_message=None
        # emd=re.Reentrancy_detection(AA)
        # edge_analysis,log_message=emd.Reentrancy_detect(num)
        print("Detection finished")
        print("-------------------------")

        return AA,edge_analysis,log_message,search_fe

    def cfg_output(self,AA,n,log,edge_analysis):
        print("Ready to ouput to a folder ")
        json_result={'contract':'',
                     'bytecode':'',
                     'runtime bytecode':'',
                     'number of basic blocks':'',
                     'block index': '',
                     'cfg': '',
                    }
        json_result['bytecode']=self.bytecode
        json_result['runtime bytecode']=str(self.Runtime_bytecode)
        json_result['number of basic blocks'] = len(self.Basic_blocks)
        json_result['block index']=str(list(enumerate(self.Block_index)))
        #json_result['cfg'] = '------------------------------------'
        cfg_nodes=[]
        for i in range(len(AA.nodes)):
            cfg_node={'block number':'',
                       'index': '',
                       'length': '',
                       'easyJump': '',
                       'resolved': '',
                       }
            cfg_node['block number']=i
            cfg_node['index']=AA.block_index[i]
            cfg_node['length']=AA.nodes[i].length
            cfg_node['easyJump']=AA.nodes[i].easyJump
            cfg_node['resolved']=AA.nodes[i].resolved
            cfg_node['onlyAmode'] = AA.nodes[i].onlyAmode
            cfg_node['bytecode'] = str(AA.nodes[i].bytecode)
            cfg_nodes.append(cfg_node)

        cfg_edges=[]
        cfg_edges.append({'number of edges':len(AA.edges)})
        for i in range(len(AA.edges)):
            cfg_edge={'from':'',
                       'to':''
                       }
            cfg_edge['from']=AA.edges[i][0]
            cfg_edge['to']=AA.edges[i][1]
            cfg_edge['type'] = AA.edges[i][2]
            cfg_edges.append(cfg_edge)

        json_result['cfg']=[cfg_nodes,cfg_edges]
        #json_result['Reentrancy']=log
        #json_result['edge_analysis']=edge_analysis
        json_str=json.dumps(json_result,indent=4,separators=(',',':'))
        print(json_str)
        with open('Test Result/new_scp_re{}.json'.format(n), 'w') as f:
            f.write(json_str)



def main():
    cfg = cfg_construction(input_contract())
    #AA=cfg.cfg_construct('x')
    #cfg.cfg_output(AA,0)

    #for test
    test_bytecode,filename=input_contract11()

    for i in range(len(test_bytecode)):
        # try:
        cfg.bytecode=test_bytecode[i]#change
        AA,edge_analysis,log_message,search_fe=cfg.cfg_construct(int(filename[i]))
        cfg.cfg_output(AA, int(filename[i]),log_message,edge_analysis)
        print("complete the {} !!".format(i + 1))
        break


