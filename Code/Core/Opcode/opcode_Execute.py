import math
#from Core.Opcode import std_Opcodes as ops
#from Core.CFG import std_Opcodes
import sys
sys.path.append("..")
from ..Opcode import std_Opcodes as ops
# from CFG import std_Opcodes
def strtoint(a):
    end = 0
    if a=='unknown':
        return a
    for index in range(len(a)):
        intermediate = bytes(a[index], 'UTF-8')
        str_int = int.from_bytes(intermediate, 'big')
        if (str_int > 47 and str_int < 58):
            str_int = str_int - 48
        elif (str_int > 96 and str_int < 103):
            str_int = str_int - 87
        try :
            end=end+int(math.pow(16,len(a)-index-1)*str_int)
        except:# deal with overflow Error
            end='unknown'
    return end

def compute01(stack):
    a=strtoint(stack.pop())
    b=strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a+b
        stack.append(hex(c))
    return stack
def compute02(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a*b
        stack.append(hex(c))
    return stack
def compute03(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a-b
        stack.append(hex(c))
    return stack
def compute04(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a//b
        stack.append(hex(c))
    return stack
def compute05(stack):#SDIV TODO
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a//b
        stack.append(hex(c))
    return stack
def compute06(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a%b
        stack.append(hex(c))
    return stack
def compute07(stack):#SMOD TODO
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a%b
        stack.append(hex(c))
    return stack
def compute08(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    N = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown' or N=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=(a+b)%N
        stack.append(hex(c))
    return stack
def compute09(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    N = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown' or N=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c = (a * b) % N
        stack.append(hex(c))
    return stack
def compute0a(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=a**b
        stack.append(hex(c))
    return stack
def compute0b(stack):#SIGNXTEND TODO
    b = strtoint(stack.pop())
    x = strtoint(stack.pop())
    c = 'unknown'
    stack.append(c)
    # if (x=='unknown' or b=='unknown'):
    #     c='unknown'
    #     stack.append(c)
    # else:
    #     y=SIGNEXETEND(x,b)
    #     stack.append(hex(y))
    return stack
def compute10(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        if (a<b):
            c=1
        else:
            c=0
        stack.append(hex(c))
    return stack
def compute11(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        if (a>b):
            c=1
        else:
            c=0
        stack.append(hex(c))
    return stack
def compute12(stack):#SLT
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        if (a<b):
            c=1
        else:
            c=0
        stack.append(hex(c))
    return stack
def compute13(stack):#SGT
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        if (a>b):
            c=1
        else:
            c=0
        stack.append(hex(c))
    return stack
def compute14(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        if (a==b):
            c=1
        else:
            c=0
        stack.append(hex(c))
    return stack
def compute15(stack):
    a = strtoint(stack.pop())
    if (a=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        if (a==0):
            c=1
        else:
            c=0
        stack.append(hex(c))
    return stack
def compute16(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=hex(a&b)  #temporary treatment
        stack.append(c)
    return stack
def compute17(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c = a | b
        stack.append(hex(c))
    return stack
def compute18(stack):
    a = strtoint(stack.pop())
    b = strtoint(stack.pop())
    if (a=='unknown' or b=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c = a ^ b
        stack.append(hex(c))
    return stack
def compute19(stack):
    a = strtoint(stack.pop())
    if (a=='unknown' ):
        c='unknown'
        stack.append(c)
    else:
        c = ~ a
        stack.append(hex(c))
    return stack
def compute1a(stack):
    i=strtoint(stack.pop())
    x=strtoint(stack.pop())
    if (i=='unknown' or x=='unknown'):
        y='unknown'
        stack.append(y)
    else:
        y=(x>>(248-i*8))&0xFF
        stack.append(hex(y))
    return stack
def compute1b(stack):
    s=strtoint(stack.pop())
    v=strtoint(stack.pop())
    if (s=='unknown' or v=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=v<<s
        stack.append(hex(c))
    return stack
def compute1c(stack):
    s = strtoint(stack.pop())
    v = strtoint(stack.pop())
    if (s=='unknown' or v=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=v>>s
        stack.append(hex(c))
    return stack
def compute1d(stack):#SAR TODO
    s = strtoint(stack.pop())
    v = strtoint(stack.pop())
    if (s=='unknown' or v=='unknown'):
        c='unknown'
        stack.append(c)
    else:
        c=v>>s
        stack.append(hex(c))
    return stack
def deal_unknown(stack,left,right):
    if(left==0 and right==0):
        pass
    elif(left==0 and right>0):
        for i in range(right):
            stack.append('unknown')
    elif(left>0 and right==0):
        for i in range(left):
            stack.pop()
    else:
        for i in range(left):
            stack.pop()
        for i in range(right):
            stack.append('unknown')
    return stack

#When we execute all instrutions, we need to check the validity of these instrutions.
def op_execution(bytecode,pc):
    stack_record=[]#record the stack
    stack=[]
    final_jumptarget=[]
    whole_Length=len(bytecode)
    count=1
    only_AMode=True #set complex jump situation that EthorSolve used as mode A
    for i in bytecode:
        if len(bytecode)==113 and bytecode[1]=='610117' and i=='1b':
            print("look")
        if(i[0]=='0' or i[0]=='1'):#Stop and Arithmetic Operations
            #check mode A
            if(i[0]=='1'and i[1]=='6'):
                pass
            else:
                if i[1]!='0':
                    only_AMode=False
            if(i[1]=='0'and i[0]=='0'):#STOP
                print("stop")
            else:
                computeFun='compute'+i[0]+i[1]
                stack=globals()[computeFun](stack)
                print("")
        elif (i[0] == '2'):#Comparsion and Bitwise Logic Operations
            if(i[1]!='0'):
                only_AMode = False
            if(i[1]=='0'):#SHA3 hash=keccak256(memory[offset:offset+length])
                stack=deal_unknown(stack,2,1)
        elif (i[0] == '3'):#Environmental Information
            if (i[1] == '0' or i[1]=='2'or i[1]=='3'or i[1]=='4'or i[1]=='6'or i[1]=='8'or i[1]=='a'or i[1]=='d'):
                stack = deal_unknown(stack, 0, 1)
            elif(i[1] == '1' or i[1]=='5'or i[1]=='b'or i[1]=='f'):
                stack = deal_unknown(stack, 1, 1)
            elif(i[1] == '7' or i[1]=='9'or i[1]=='e'):
                stack = deal_unknown(stack, 3, 0)
            elif(i[1] == 'c'):
                stack = deal_unknown(stack, 4, 0)
        elif (i[0] == '4'):#Blcok Information
            if (i[1] == '1' or i[1] == '2' or i[1] == '3' or i[1] == '4' or i[1] == '5' or i[1] == '6'
                or i[1] == '7' or i[1] == '8'):
                stack = deal_unknown(stack, 0, 1)
            elif (i[1] == '0' ):
                stack = deal_unknown(stack, 1, 1)
        elif (i[0] == '5'):#Stack, Memory, Storage and Flow Operations
            # check mode A
            if (i[1] == '8'):#pc
               only_AMode=False
            if (i[1] == '0'):#POP
                stack.pop()
            elif (i[1] == '1' or i[1]=='4'):
                stack = deal_unknown(stack, 1, 1)
            elif (i[1] == '2' or i[1]=='3'or i[1]=='5'):
                stack = deal_unknown(stack, 2, 0)
            elif (i[1]=='9'or i[1]=='a'):
                stack = deal_unknown(stack, 0, 1)
            elif (i[1] == '8'):#PC
                print("PC IN NEED TODO")
                stack.append("pccccccccc")
            elif(i[1]=='6'):#JUMP don't need to special handling
                if(count==whole_Length):
                    final_jumptarget.append(stack.pop())
                else:
                    stack.pop()
            elif (i[1] == '7'):  # JUMPI has two target addresses, but we only concern the first jump target address
                if (count == whole_Length):
                    cond=stack.pop()
                    final_jumptarget.append(stack.pop())
                else:#Since the 'bytecode' consists of several blocks, we don't need to deal with medium blocks
                    stack.pop()
                    stack.pop()
        elif (i[0] == '6' or i[0]=='7'):#PUSH Operation
            stack.append(i[2:])
        elif(i[0]=='8'):#DUP
            dupdepth=strtoint(i[1])
            dupdata=stack[(-1)-(dupdepth)]
            stack.append(dupdata)
        elif(i[0]=='9'):#SWAP
            swapdepth=strtoint(i[1])
            swapdata1=stack[(-2)-(swapdepth)]
            swapdata2=stack[-1]
            stack[-1]=swapdata1
            stack[(-2)-(swapdepth)]=swapdata2
        elif(i[0]=='a'):#LOG
            depth=int(i[1])+2
            for index in range(depth):
                stack.pop()
        elif(i[0]=='f'):#System Operation
            if i[1]=='0':#f0 create
                deal_unknown(stack, 3, 1)
            elif i[1]=='1' or i[1]=='2':
                deal_unknown(stack,7,1)
            elif i[1]=='3' or i[1]=='d' or i[1]=='f' :#return cannot continue to execute the program
                return [],only_AMode
            elif i[1]=='4'or i[1]=='a':
                deal_unknown(stack,6,1)
            elif i[1]=='5':
                deal_unknown(stack,4,1)
        count=count+1
        temp=stack.copy()
        stack_record.append(temp)
        print(stack)
    return final_jumptarget,only_AMode

# bytecode=[['0','1'],['0','2']]
# ddd=['5b', '610059', '6100ed', '56','5b', '6000', '80', '6000', '81', '54', '80', '92', '91', '90', '6001', '01', '91',
#      '90', '50', '55', '50', '60aa', '6000', '54', '14', '15', '610115', '57', '6000', '54', '90', '50', '61011a', '56'
#     , '5b', '90', '56']
# currentpc=[81,237,268,282]
#
# op_execution(ddd,currentpc)
#When we execute all instrutions, we need to check the validity of these instrutions.
#this functions designs
def op_execution2(bytecode):
    stack_record=[]#record the stack
    stack=[]
    count=1
    only_AMode=True #set complex jump situation that EthorSolve used as mode A
    for i in bytecode:
        try:
            if(i[0]=='0' or i[0]=='1'):#Stop and Arithmetic Operations
                #check mode A
                if(i[0]=='1'and i[1]=='6'):
                    pass
                else:
                    if i[1]!='0':
                        only_AMode=False
                if(i[1]=='0'and i[0]=='0'):#STOP
                    print("stop")
                else:
                    computeFun='compute'+i[0]+i[1]
                    stack=globals()[computeFun](stack)
            elif (i[0] == '2'):#Comparsion and Bitwise Logic Operations
                if(i[1]!='0'):
                    only_AMode = False
                if(i[1]=='0'):#SHA3 hash=keccak256(memory[offset:offset+length])
                    stack=deal_unknown(stack,2,1)
            elif (i[0] == '3'):#Environmental Information
                if (i[1] == '0' or i[1]=='2'or i[1]=='3'or i[1]=='4'or i[1]=='6'or i[1]=='8'or i[1]=='a'or i[1]=='d'):
                    stack = deal_unknown(stack, 0, 1)
                elif(i[1] == '1' or i[1]=='5'or i[1]=='b'or i[1]=='f'):
                    stack = deal_unknown(stack, 1, 1)
                elif(i[1] == '7' or i[1]=='9'or i[1]=='e'):
                    stack = deal_unknown(stack, 3, 0)
                elif(i[1] == 'c'):
                    stack = deal_unknown(stack, 4, 0)
            elif (i[0] == '4'):#Blcok Information
                if (i[1] == '1' or i[1] == '2' or i[1] == '3' or i[1] == '4' or i[1] == '5' or i[1] == '6'
                    or i[1] == '7' or i[1] == '8'):
                    stack = deal_unknown(stack, 0, 1)
                elif (i[1] == '0' ):
                    stack = deal_unknown(stack, 1, 1)
            elif (i[0] == '5'):#Stack, Memory, Storage and Flow Operations
                # check mode A
                if (i[1] == '8'):#pc
                   only_AMode=False
                if (i[1] == '0'):#POP
                    stack.pop()
                elif (i[1] == '1' or i[1]=='4'):
                    stack = deal_unknown(stack, 1, 1)
                elif (i[1] == '2' or i[1]=='3'or i[1]=='5'):
                    stack = deal_unknown(stack, 2, 0)
                elif (i[1]=='9'or i[1]=='a'):
                    stack = deal_unknown(stack, 0, 1)
                elif (i[1] == '8'):#PC
                    print("PC IN NEED TODO")
                    stack.append("pccccccccc")
                elif(i[1]=='6'):#JUMP don't need to special handling
                        stack.pop()
                elif (i[1] == '7'):  # JUMPI has two target addresses, but we only concern the first jump target address
                        stack.pop()
                        stack.pop()
            elif (i[0] == '6' or i[0]=='7'):#PUSH Operation
                stack.append(i[2:])
            elif(i[0]=='8'):#DUP
                dupdepth=strtoint(i[1])
                dupdata=stack[(-1)-(dupdepth)]
                stack.append(dupdata)
            elif(i[0]=='9'):#SWAP
                swapdepth=strtoint(i[1])
                swapdata1=stack[(-2)-(swapdepth)]
                swapdata2=stack[-1]
                stack[-1]=swapdata1
                stack[(-2)-(swapdepth)]=swapdata2
            elif(i[0]=='a'):#LOG
                depth=int(i[1])+2
                for index in range(depth):
                    stack.pop()
            elif(i[0]=='f'):#System Operation
                if i[1]=='0':#f0 create
                    deal_unknown(stack, 3, 1)
                elif i[1]=='1' :# check UNSAFE CALL
                    if stack!=[] and stack[-2]!='unknown':
                        deal_unknown(stack,7,1)
                        return stack,False
                    else:
                        return [],True
                elif i[1]=='2':
                    deal_unknown(stack,7,1)
                elif i[1]=='3' or i[1]=='d' or i[1]=='f' :#return cannot continue to execute the program
                    return [],only_AMode
                elif i[1]=='4'or i[1]=='a':
                    deal_unknown(stack,6,1)
                elif i[1]=='5':
                    deal_unknown(stack,4,1)
        except IndexError:
            print("wrong")
            return [],True
        count=count+1
        temp=stack.copy()
        stack_record.append(temp)
        #print(stack)
    #if executes successful, it implies that there is no unsafe call
    return stack,False

#For features extraction
def op_extract(bytecode,pc):
    stack_record=[]#record the stack
    stack=[]
    count=0
    only_AMode=True #set complex jump situation that EthorSolve used as mode A
    #At first,we fill some 'unknown's into stack to avoid overflow.
    for i in range(20):
        stack.append('unknown')
    for i in bytecode:
        if count==len(bytecode)-1:
            if(stack[-1]=='00'):
                return True
            else:
                return False
        if(i[0]=='0' or i[0]=='1'):#Stop and Arithmetic Operations
            #check mode A
            if(i[0]=='1'and i[1]=='6'):
                pass
            else:
                if i[1]!='0':
                    only_AMode=False
            if(i[1]=='0'and i[0]=='0'):#STOP
                print("stop")
            else:
                computeFun='compute'+i[0]+i[1]
                stack=globals()[computeFun](stack)
                print("")
        elif (i[0] == '2'):#Comparsion and Bitwise Logic Operations
            if(i[1]!='0'):
                only_AMode = False
            if(i[1]=='0'):#SHA3 hash=keccak256(memory[offset:offset+length])
                stack=deal_unknown(stack,2,1)
        elif (i[0] == '3'):#Environmental Information
            if (i[1] == '0' or i[1]=='2'or i[1]=='3'or i[1]=='4'or i[1]=='6'or i[1]=='8'or i[1]=='a'or i[1]=='d'):
                stack = deal_unknown(stack, 0, 1)
            elif(i[1] == '1' or i[1]=='5'or i[1]=='b'or i[1]=='f'):
                stack = deal_unknown(stack, 1, 1)
            elif(i[1] == '7' or i[1]=='9'or i[1]=='e'):
                stack = deal_unknown(stack, 3, 0)
            elif(i[1] == 'c'):
                stack = deal_unknown(stack, 4, 0)
        elif (i[0] == '4'):#Blcok Information
            if (i[1] == '1' or i[1] == '2' or i[1] == '3' or i[1] == '4' or i[1] == '5' or i[1] == '6'
                or i[1] == '7' or i[1] == '8'):
                stack = deal_unknown(stack, 0, 1)
            elif (i[1] == '0' ):
                stack = deal_unknown(stack, 1, 1)
        elif (i[0] == '5'):#Stack, Memory, Storage and Flow Operations
            # check mode A
            if (i[1] == '8'):#pc
               only_AMode=False
            if (i[1] == '0'):#POP
                stack.pop()
            elif (i[1] == '1' or i[1]=='4'):
                stack = deal_unknown(stack, 1, 1)
            elif (i[1] == '2' or i[1]=='3'or i[1]=='5'):
                stack = deal_unknown(stack, 2, 0)
            elif (i[1]=='9'or i[1]=='a'):
                stack = deal_unknown(stack, 0, 1)
            elif (i[1] == '8'):#PC
                print("PC IN NEED TODO")
                stack.append("pccccccccc")
            elif(i[1]=='6'):#JUMP don't need to special handling
                stack.pop()
            elif (i[1] == '7'):  # JUMPI has two target addresses, but we only concern the first jump target address
                stack.pop()
                stack.pop()
        elif (i[0] == '6' or i[0]=='7'):#PUSH Operation
            stack.append(i[2:])
        elif(i[0]=='8'):#DUP
            dupdepth=strtoint(i[1])
            dupdata=stack[(-1)-(dupdepth)]
            stack.append(dupdata)
        elif(i[0]=='9'):#SWAP
            swapdepth=strtoint(i[1])
            swapdata1=stack[(-2)-(swapdepth)]
            swapdata2=stack[-1]
            stack[-1]=swapdata1
            stack[(-2)-(swapdepth)]=swapdata2
        elif(i[0]=='a'):#LOG
            depth=int(i[1])+2
            for index in range(depth):
                stack.pop()
        elif(i[0]=='f'):#System Operation
            if i[1]=='0':#f0 create
                deal_unknown(stack, 3, 1)
            elif i[1]=='1' or i[1]=='2':
                deal_unknown(stack,7,1)
            elif i[1]=='3' or i[1]=='d' or i[1]=='f' :#return cannot continue to execute the program
                return [],only_AMode
            elif i[1]=='4'or i[1]=='a':
                deal_unknown(stack,6,1)
            elif i[1]=='5':
                deal_unknown(stack,4,1)
        count=count+1
        temp=stack.copy()
        stack_record.append(temp)
        print(stack)

def timer(stack,left,right):
    if(left==0 and right==0):
        pass
    elif(left==0 and right>0):
        for i in range(right):
            stack.append('unknown')
    elif(left>0 and right==0):
        for i in range(left):
            stack.pop()
    else:
        check=False
        for i in range(left):
            x=stack.pop()
            if x=='timestamp':
                check=True
        if check==True:
            for i in range(right):
                stack.append('timestamp')
        else:
            for i in range(right):
                stack.append('unknown')
    return stack
#For features extraction:timestamp
def op_extract2(bytecode,pc):
    stack_record=[]#record the stack
    stack=[]
    count=0
    only_AMode=True #set complex jump situation that EthorSolve used as mode A
    #At first,we fill some 'unknown's into stack to avoid overflow.
    for i in range(20):
        stack.append('unknown')
    for i in bytecode:
        left = ops.standard_opcodes['0x' + i[:2]][1]
        right = ops.standard_opcodes['0x' + i[:2]][2]
        if i=='42':
            stack.append('timestamp')
            continue
        if(i[0]=='0' or i[0]=='1' or i[0]=='2' or i[0] =='3' or i[0]=='4'):#Stop and Arithmetic Operations
            stack=timer(stack,left,right)

        elif (i[0] == '5'):#Stack, Memory, Storage and Flow Operations
            # check mode A
            if (i[1] == '8' or i[1] == '1' or i[1]=='4' or i[1] == '2' or i[1]=='3'or i[1]=='5' or i[1]=='9'or i[1]=='a' or i[1] == '8'):
               stack=timer(stack,left,right)
            if (i[1] == '0'):#POP
                stack.pop()
            elif(i[1]=='6'):#JUMP don't need to special handling
                stack.pop()
                return False
            elif (i[1] == '7'):  # JUMPI has two target addresses, but we only concern the first jump target address
                x1=stack.pop()
                x2=stack.pop()
                if x1=='timestamp' or x2=='timestamp':
                    return True
                else:
                    return False
        elif (i[0] == '6' or i[0]=='7'):#PUSH Operation
            stack.append(i[2:])
        elif(i[0]=='8'):#DUP
            dupdepth=strtoint(i[1])
            dupdata=stack[(-1)-(dupdepth)]
            stack.append(dupdata)
        elif(i[0]=='9'):#SWAP
            swapdepth=strtoint(i[1])
            swapdata1=stack[(-2)-(swapdepth)]
            swapdata2=stack[-1]
            stack[-1]=swapdata1
            stack[(-2)-(swapdepth)]=swapdata2
        elif(i[0]=='a'):#LOG
            depth=int(i[1])+2
            for index in range(depth):
                stack.pop()
        elif(i[0]=='f'):#System Operation
            if i[1]=='3' or i[1]=='d' or i[1]=='f' :#return cannot continue to execute the program
                return [],only_AMode
            else:
                stack=timer(stack,left,right)
        count=count+1
        temp=stack.copy()
        stack_record.append(temp)
        print(stack)