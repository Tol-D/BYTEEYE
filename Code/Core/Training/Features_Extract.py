import numpy as np
import re
from ..Opcode import opcode_Execute as oe
ArithmeticF=['01','02','03','04','05','06','07','08','09','0a','0b','10','11','12','13',
             '14','15','16','17','18','19','1a','1b','1c','1d']
MemoryF=['51','52','53','54','55']
CallF=['f1','f4','fa']
CompareF=['10','11','12','13','14']
exitF=['56','57','fd','ff','f3','00'] #jump, jumpi, revert, return, selfdestruct, stop
arithF=['01','02','03','04','05','0a']#add, mul, sub, div, sdiv, exp


# def count_offspring(block,edges):
#     points=[block['index']]
#     toe=len(edges)
#     for i in range(1,len(edges)):
#         if edges[i]['from'] in points and edges[i]['to'] not in points:
#             points.append(edges[i]['to'])
#     #related destinations not equal to child blocks!
#     if len(points)>=(0.75*toe):
#         return 3
#     elif len(points)>=(0.5*toe):
#         return 2
#     elif len(points)>=(0.25*toe):
#         return 1
#     if len(points)<(0.25*toe):
#         return 0
def count_offspring(block,edges):
    points=[block['index']]
    for i in range(1,len(edges)):
        if edges[i]['from'] in points and edges[i]['to'] not in points:
            points.append(edges[i]['to'])
    #related destinations not equal to child blocks!
    return (len(points))


#feature 'access control'
def rule_access(block,edges):
    judge1 = False
    judge2 = False
    ra=0
    for t in block['bytecode'].split('\''):
        if t=='33':#CALLER
            judge1=True
            continue
        if judge1 and t=='14':#EQ
            judge2=True
            continue
        if judge1 and judge2 and t=='fd':#REVERT
            ra=1
        if judge1 and judge2 and t=='57':#JUMPI
            ra=1
    return ra

def rule_SUB1(block,edges):
    judge1 = False
    j_data=False
    ra=0
    #Bytecode list
    t1 = re.split(':|,|\'', block['bytecode'])
    t2=[]
    for t in range(len(t1)):
        if t1[t]=='[' or t1[t]=='' or t1[t]==' ' or t1[t]==']':
            continue
        t2.append(t1[t])
    index_push=[]
    index_store=0
    index=0
    #check whether has potential dependency or not
    for t in t2:
        if t=='6000':#PUSH0
            judge1=True
            index_push.append(index)
            index=index+1
            continue
        if judge1 and t=='55':#SSTORE
            index_store=index
            j_data=True
            break
        index=index+1
    if (j_data):
        if((oe.op_extract(t2[index_push[-1]:index_store+1],0))==False):
            if(oe.op_extract(t2[0:index_store+1],0)):
                ra=1
        else:
            ra=1
    return ra

def rule_SUB2(block,edges):
    judge1 = False
    judge2 = False
    ra = 0
    for t in block['bytecode'].split('\''):
        if t == '54':  # SLOAD
            judge1 = True
            continue
        if judge1 and t == '03':  # SUB
            judge2 = True
            continue
        if judge1 and judge2 and t == '55':  # SSTORE
            ra = 1
    return ra
#feature timestamp
def time_stamp(block,edges,blockn=[]):
    #preprocess
    judge=False
    if blockn==[]:
        test=block['bytecode'].split('\'')
    else:
        test=(block['bytecode']+'\''+blockn['bytecode']).split('\'')
    for i in range(len(test)-1,-1,-1):
        if test[i]==', ' or test[i]=='[' or test[i]==']':
            del test[i]
    #Rule out
    if test[-1]=='56':#JUMP
        return 0
    exist=False
    for t in test:
        if t in CompareF:#LT,GT,...
            exist=True
    if exist==False:
        return 0
    #check timestamp dependency
    count=0
    for t in test:
        if t=='42':#timestamp
            result=oe.op_extract2(test[count:], 0)
            if result==True:
                return 1
        count=count+1
    # for t in test:
    #     if t=='42':#timestamp
    #         judge=True
    #         cc=0
    #         continue
    #     if judge and cc<2:#timestamp becomees the comparison object
    #         if t=='10' or t=='11':
    #             return 1
    #         cc=cc+1
    return 0

#feature integer overflow
def overflow(test):
    io=0
    judge1,judge2=False,False
    for t in test:
        if t == '54':  # SLOAD
            judge1 = True
            continue
        if judge1 and t in arithF:  # SUB...
            judge2 = True
            continue
        if judge1 and judge2 and t == '55':  # SSTORE
            io = 1
    return io

def overflow_protect(test,blockn=[]):
    p=0
    if blockn!=[]:
        test2=(blockn['bytecode']).split('\'')
        for i in range(len(test2)-1,-1,-1):
            if test2[i]==', ' or test2[i]=='[' or test2[i]==']':
                del test2[i]
        for i in range(len(test2)):
            test.append(test2[i])

    judge1, judge2,judge3 = False, False,False
    for t in test:
        if t == '10' or t == '11':  # LT GT
            judge1 = True
            sc = 0
            continue
        if judge1 and t == '15':  # two or three ISZERO
            sc = sc + 1
            if sc > 1:
                judge2 = True
            continue
        if judge1 and judge2 and t == '57':  # JUMPI
            judge3=True
            continue
        if (t=='fd' or t=='fe')  and judge1 and judge2 and judge3:
            p = 1
    return p
#main features
def Eight_features(maps,i):
    # acquire features  n*8
    n = len(maps[i][0])  # n denotes the number of nodes
    feature = np.zeros((n, 9), dtype=np.float32)  # feature matrix
    havef=False
    for j in range(n):
        test = maps[i][0][j]['bytecode'].split('\'')
        for t in range(len(test) - 1, -1, -1):
            if test[t] == ', ' or test[t] == '[' or test[t] == ']':
                del test[t]
        if 'f1' in test:
            havef=True
    for j in range(n):
        test = maps[i][0][j]['bytecode'].split('\'')
        for t in range(len(test) - 1, -1, -1):
            if test[t] == ', ' or test[t] == '[' or test[t] == ']':
                del test[t]

        #Arithmetic Features
        for t in test:
            if t in ArithmeticF:
               feature[j][0] = feature[j][0]+1
        # Memory Features
        for t in test:
            if t in MemoryF:
               feature[j][1] = feature[j][1]+1
        # Call Features
        for t in test:
            if t in CallF:
               feature[j][2] = feature[j][2]+1

        #Count offspring
        feature[j][3]=count_offspring(maps[i][0][j],maps[i][1])

        #F1 CALL related to Reentrancy
        if 'f1'in maps[i][0][j]['bytecode'].split('\''):
            feature[j][4]=1
        #55 SSTORE  related to Reentrancy
        if havef:
            if '55' in maps[i][0][j]['bytecode'].split('\''):
                feature[j][5] = feature[j][5]+1

        # #Timestamp
        # if '42'in test:
        #     if test[-1] not in exitF and j<n-1:#special
        #         feature[j][4] = time_stamp(maps[i][0][j], maps[i][1],maps[i][0][j+1])
        #     else:
        #         feature[j][4]=time_stamp(maps[i][0][j],maps[i][1])
        # if 'f1' in test:
        #     feature[j][5]=feature[j][5]+1

        #'access control' feature without data dependency
        feature[j][6] = rule_access(maps[i][0][j], maps[i][1])
        # 'SUB1' feature without data dependency
        feature[j][7] = rule_SUB1(maps[i][0][j], maps[i][1])
        # 'SUB2' feature without data dependency
        feature[j][8] = rule_SUB2(maps[i][0][j], maps[i][1])

        # #overflow
        #new
        # oper=False
        # for x in range(len(test)):
        #     if test[x] in arithF:
        #         oper=True
        # #if '01'in arithF:  #write wrong
        # if oper:
        #     feature[j][4]=overflow(test)
        # if '35' in maps[i][0][j]['bytecode'].split('\''):
        #     feature[j][5]=feature[j][5]+1
        # if '01'in test or '02'in test or '03'in test:
        #     feature[j][6]=feature[j][6]+1
        # feature[j][7] = rule_access(maps[i][0][j], maps[i][1])
        # if j<n-1:
        #     feature[j][8] = overflow_protect(test,maps[i][0][j+1])
        # else:
        #     feature[j][8] = overflow_protect(test)

    return feature
#
# def Eight_features(maps,i):
#     # acquire features  n*8
#     n = len(maps[i][0])  # n denotes the number of nodes
#     feature = np.zeros((n, 9), dtype=np.float32)  # feature matrix
#     havef=False
#     for j in range(n):
#         if 'f1' in maps[i][0][j]['bytecode'].split('\''):
#             havef=True
#     for j in range(n):
#         #Arithmetic Features
#         for t in maps[i][0][j]['bytecode'].split('\''):
#             if t in ArithmeticF:
#                feature[j][0] = feature[j][0]+1
#         # Memory Features
#         for t in maps[i][0][j]['bytecode'].split('\''):
#             if t in MemoryF:
#                feature[j][1] = feature[j][1]+1
#         # Call Features
#         for t in maps[i][0][j]['bytecode'].split('\''):
#             if t in CallF:
#                feature[j][2] = feature[j][2]+1
#         #Count offspring
#         feature[j][3]=count_offspring(maps[i][0][j],maps[i][1])
#         # #F1 CALL related to Reentrancy
#         # if 'f1'in maps[i][0][j]['bytecode'].split('\''):
#         #     feature[j][4]=1
#         # #55 SSTORE  related to Reentrancy
#         # if havef:
#         #     if '55' in maps[i][0][j]['bytecode'].split('\''):
#         #         feature[j][5] = feature[j][5]+1
#         # Timestamp
#         # if '42'in maps[i][0][j]['bytecode'].split('\''):
#         #     feature[j][4]=feature[j][4]+1
#         # if 'f1' in maps[i][0][j]['bytecode'].split('\''):
#         #     feature[j][5]=feature[j][5]+1
#         #Ovreflow
#         if '01'in maps[i][0][j]['bytecode'].split('\'') or \
#                 '02'in maps[i][0][j]['bytecode'].split('\'') or \
#                 '03'in maps[i][0][j]['bytecode'].split('\''):
#             feature[j][4]=feature[j][4]+1
#         # if 'f1' in maps[i][0][j]['bytecode'].split('\''):
#         #     feature[j][5]=feature[j][5]+1
#
#         #'access control' feature without data dependency
#         feature[j][6] = rule_access(maps[i][0][j], maps[i][1])
#         # 'SUB1' feature without data dependency
#         feature[j][7] = rule_SUB1(maps[i][0][j], maps[i][1])
#         # 'SUB2' feature without data dependency
#         feature[j][8] = rule_SUB2(maps[i][0][j], maps[i][1])
#
#     return feature