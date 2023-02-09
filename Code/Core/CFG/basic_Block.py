import sys
sys.path.append("..")
from ..Opcode import std_Opcodes as ops
#from . import std_Opcodes as ops
class BasicBlock:
    def __init__(self,bytecode,index,start,end):
        self.info=[]
        self.index=index
        self.length=end-start+1
        self.bytecode=bytecode[start:(end+1)]
        self.startStack=0 # to solve the complex jump
        self.addStack=0 # to solve the complex jump
        self.easyJump=False #calssify specific types
        self.resolved=False #calssify specific types
        self.onlyAmode=True #calssify specific types
        self.edges=[]
        self.Revedges=[]
    def compute_StartStack(self):
        count_left=0
        count_right=0
        for i in range(len(self.bytecode)):
            if(self.bytecode[i][0:2]=='fe'):
                continue
            temp_op='0x'+self.bytecode[i][0:2]
            count_left = count_left+ops.standard_opcodes[temp_op][1]
            count_right = count_right + ops.standard_opcodes[temp_op][2]
        self.addStack=count_right-count_left#the increment after executing the whole block
        #compute the requirment stack depth before executing the whole block
        current_stk=0
        stk =0
        for i in range(len(self.bytecode)):
            if(self.bytecode[i][0:2]=='fe'):
                continue
            temp_op = '0x' + self.bytecode[i][0:2]
            if (current_stk >= ops.standard_opcodes[temp_op][1]):
                current_stk = current_stk + (ops.standard_opcodes[temp_op][2] - ops.standard_opcodes[temp_op][1])
            else:
                stk = stk + (ops.standard_opcodes[temp_op][1] - current_stk)
                current_stk = ops.standard_opcodes[temp_op][2]
        self.startStack=stk

        # self.startStack = stk
        # if(stk>0):#Variables apparently need to execute more routes
        #     for i in range(len(self.bytecode)):
        #         temp_op = '0x' + self.bytecode[i][0:2]
        #         if(current_stk>=ops.standard_opcodes[temp_op][1]):
        #             current_stk=current_stk+(ops.standard_opcodes[temp_op][2]-ops.standard_opcodes[temp_op][1])
        #         else:
        #             stk=stk+(ops.standard_opcodes[temp_op][1]-current_stk)
        #             current_stk=ops.standard_opcodes[temp_op][2]
        #     self.startStack=stk
        # else:
        #     current_stk = 0
        #     addV=0
        #     #These variables also need to be checked that whether they satisfy conditions or not
        #     for i in range(len(self.bytecode)):
        #         temp_op = '0x' + self.bytecode[i][0:2]
        #         if(current_stk>=ops.standard_opcodes[temp_op][1]):#general situation
        #             current_stk=current_stk+(ops.standard_opcodes[temp_op][2]-ops.standard_opcodes[temp_op][1])
        #         else:
        #             addV=addV+(ops.standard_opcodes[temp_op][1]-current_stk)
        #             current_stk=ops.standard_opcodes[temp_op][2]
        #     if addV!=0:
        #         self.startStack = addV
        #     else:
        #         self.startStack=stk