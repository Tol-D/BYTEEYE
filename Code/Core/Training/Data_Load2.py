#loading data from ESC dataset  Reentrancy
import os
import re
import csv
import numpy as np
import scipy.sparse as sp
import math
import json
import torch
import random
from . import Features_Extract as fee

#The Third step (Analogy to Data_load.py )
#read data from testRE{} files
def read_data():
    maps=[]
    ############################################################################
    #new scp_reentrancy
    for count in range(541):
        try:
            with open('Test Result/new_scp_re{}.json'.format(count), 'r') as f:
                a = json.load(f)
        except:  # if cannot find relating files
            pass
        map=a['cfg']
        maps.append(map)

    # before=0
    # for i in range(len(maps)):
    #     before=before+len(maps[i][1])-1
    # #for no Complex jumps
    # for i in range(len(maps)):
    #     for j in range(len(maps[i][1])-1,0,-1):
    #         if maps[i][1][j]['type']!='EA':
    #             del maps[i][1][j]
    #     maps[i][1][0]['number of edges']=len(maps[i][1])-1
    # after=0
    # for i in range(len(maps)):
    #     after=after+len(maps[i][1])-1
    return maps


def load_data():
    maps=read_data()
    adjs=[]
    features=[]
    numbers=[]
    for i in range(len(maps)):
        print('now is  {:04d}'.format(i))

        n = len(maps[i][0])  # n denotes the number of nodes
        m = maps[i][1][0]['number of edges']
        numbers.append(n)
        '''if maps[i][0][-1]['bytecode']=='[\'fe\']':#we need to ensure that the final block has edges with other blocks
            n=n-1'''
        indexs=[]
        for j in range(n):
            indexs.append(maps[i][0][j]['index'])
        #acquire adjacency matrix  n*n
        adj=np.zeros((n,n),dtype=np.float32)#adjacency matrix
        for j in range(1,m+1):
            # print('now j is  {:04d}'.format(j))
            # if j==458:
            #     print("458")
            try:
                adj[indexs.index(maps[i][1][j]['from'])][indexs.index(maps[i][1][j]['to'])]=1
                adj[indexs.index(maps[i][1][j]['to'])][indexs.index(maps[i][1][j]['from'])] = 1
            except ValueError:
                fromx=0
                tox=0
                outlimit=False
                for t in range(len(indexs)):
                    if t==len(indexs)-1:
                        outlimit=True
                        break
                    if maps[i][1][j]['from']==indexs[t]:
                        fromx=t
                        break
                    if maps[i][1][j]['from']>indexs[t] and maps[i][1][j]['from']<indexs[t+1]and t<len(indexs)-1:
                        fromx=t
                        break
                for t in range(len(indexs)):
                    if t==len(indexs)-1:
                        outlimit=True
                        break
                    if maps[i][1][j]['to']==indexs[t]:
                        tox=t
                        break
                    if maps[i][1][j]['to']>indexs[t] and maps[i][1][j]['to']<indexs[t+1] and t<len(indexs)-1:
                        tox=t
                        break
                if outlimit:
                    break
                else:
                    adj[fromx][tox]=1
                    adj[tox][fromx] = 1

        #for GAT MODEL, we need new adj=adj+I
        for j in range(n):
            adj[j][j]=adj[j][j]+1

        #acquire symmetric adjacency matrix
        D=[]#D -1/2
        for j in range(n):
            temp=0
            for k in range(n):
                temp=temp+adj[j][k]
            if temp==0:#special situation
                adj[j][j+1]=1
                temp=1
            temp=1/math.sqrt(temp)
            D.append(temp)
        for j in range(n):#D^(-1/2)*adj*D^(-1/2)
            for k in range(n):
                adj[j][k]=D[j]*adj[j][k]*D[k]
        if i==4:
            print()
        feature=fee.Eight_features(maps,i)

        #Normalize matrix
        feature=normalize(feature)
        features.append(feature)
        adj=normalize(adj+sp.eye(adj.shape[0]))
        adjs.append(adj)
        # adjs.append(torch.Tensor(adj))
        # features.append(torch.Tensor(feature))

    # adjs=torch.stack(adjs,0)
    # features=torch.stack(features,0)
    print("start padding")
    #adjs,features=padding(adjs,features)
    #print(adjs[9])
    #we need to load some values because of the huge computation
    np.save('data/internal_result/adjs_scpre.npy',adjs)
    np.save('data/internal_result/features_scpre.npy', features)
    np.save('data/internal_result/numbers_scpre.npy', numbers)

    return adjs,features,numbers


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def padding(adjs,features):
    X,Y=[],[]
    max_m = 0
    for i in range(len(adjs)):
        if adjs[i].shape[0] > max_m:
            max_m = adjs[i].shape[0]
    for i in range(len(features)):
        feature=np.pad(features[i],((0,max_m-adjs[i].shape[0]),(0,0)),'constant',constant_values=(0,0))
        X.append(torch.Tensor(feature))
    for i in range(len(adjs)):
        adj=np.pad(adjs[i],((0,max_m-adjs[i].shape[0]),(0,max_m-adjs[i].shape[0])),'constant',constant_values=(0,0))
        Y.append(torch.Tensor(adj))
    features=torch.stack(X)
    adjs=torch.stack(Y)
    return adjs,features


def load_datasets():
    Labels = []

    ############################################################################
    # new scp_reentrancy
    file=open('Core/Input/contracts_Compiled/scpre_label.txt',"r")
    for line in file:
        Labels.append(int(line.strip('\n')))
    idx_train=433
    idx_valid=248
    idx_test=541

    #scp_timestamp
    # # from SCP dataset
    # file=open("Core/Input/Dataset_SB/Label_SBT.txt","r")
    # for line in file:
    #     Labels.append(int(line.strip('\n')))
    # idx_train=433
    # idx_valid=248
    # idx_test=288

    # from scp_integer overflow dataset
    # file=open("Core/Input/Dataset_SB/scpio_label.txt","r")
    # for line in file:
    #     Labels.append(int(line.strip('\n')))
    #
    # idx_train=433
    # idx_valid=248
    # idx_test=316

    return Labels,idx_train,idx_valid,idx_test

def re_balance(adjs, features,labels,idx_train,idx_valid,idx_test):
    scale=0.26
    plabel_num=71
    p_ex=int(scale*idx_train)+1#the train dataset should have "p_ex" labels.

    #compute the number of  postive labels in train dataset
    count=0
    decount=0
    for i in labels:
        if i ==1:
           count=count+1
        if i ==0:
            decount=decount+1
        if decount+count==idx_train:
            break
    # compute the number of  postive labels in valid dataset
    countv=0
    for i in range(idx_train,idx_valid):
        if labels[i]==1:
            countv=countv+1
    #This change is only adaptive to the specific datatset
    diff=p_ex-count
    diff2=countv-7
    diff3=diff-diff2
    if diff>0:
        #find examples that denote the label 1 in test dataset
        test_point=[]
        exchange_a_tests=[]
        exchange_f_tests=[]
        rand=0
        while(len(test_point)!=diff3):
            for i in range(idx_valid,idx_test,2):
                if i +1!=idx_test:
                    i=i+rand
                if labels[i]==1 and i not in test_point:
                    if len(test_point)==diff3:
                        break
                    test_point.append(i)
                    exchange_a_tests.append(adjs[i])
                    exchange_f_tests.append(features[i])
            rand=rand+1

        #valid dataset
        valid_point=[]
        exchange_a_valid = []
        exchange_f_valid = []
        rand=0
        while(len(valid_point)!=diff2):
            for i in range(idx_train,idx_valid,2):
                if i + 1!=idx_valid:
                    i=i+rand
                if labels[i]==1 and i not in valid_point:
                    if len(valid_point)==diff2:
                        break
                    valid_point.append(i)
                    exchange_a_valid.append(adjs[i])
                    exchange_f_valid.append(features[i])
            rand=rand+1

        #train dataset
        train_point=[]
        exchange_a_train = []
        exchange_f_train = []
        for i  in range(0,idx_train,3):
            if labels[i] == 0:
                train_point.append(i)
                exchange_a_train.append(adjs[i])
                exchange_f_train.append(features[i])
            if len(train_point) == diff:
                break

        #exchange value
        for i in range(len(test_point)):
            adjs[test_point[i]]=exchange_a_train[i]
            features[test_point[i]] = exchange_f_train[i]
            labels[test_point[i]]=0
        for i in range(len(valid_point)):
            adjs[valid_point[i]]=exchange_a_train[diff3+i]
            features[valid_point[i]] = exchange_f_train[diff3+i]
            labels[valid_point[i]]=0
        for i in range(len(train_point)):
            if i<diff3:
                adjs[train_point[i]]=exchange_a_tests[i]
                features[train_point[i]] = exchange_f_tests[i]
                labels[train_point[i]]=1
            else:
                adjs[train_point[i]]=exchange_a_valid[i-diff3]
                features[train_point[i]] = exchange_f_valid[i-diff3]
                labels[train_point[i]]=1

    count1 = 0
    for i in range(idx_train, idx_valid):
        if labels[i] == 1:
            count1 = count1 + 1
    count2 = 0
    for i in range(idx_valid,idx_test):
        if labels[i] == 1:
            count2 = count2 + 1
    count3 = 0
    for i in range(0, idx_train):
        if labels[i] == 1:
            count3 = count3 + 1

    return adjs, features,labels

def kfolds(labels):
    find1=[]
    find0=[]
    for i in range(len(labels)):
        if labels[i]==1:
            find1.append(i)
        if labels[i]==0:
            find0.append(i)
    #new add
    random.shuffle(find1)
    random.shuffle(find0)
    #5-folds cross validation
    #label 1
    id1,id2,id3,id4,id5=[],[],[],[],[]
    id=[id1,id2,id3,id4,id5]
    for i in range(0,len(find1),5):
        if i ==int(len(find1)/5)*5 and len(find1)%5!=0:
            t=len(find1)%5
            for j in range(t):
                id[j].append(find1[i+j])
            break
        id1.append(find1[i])
        id2.append(find1[i+1])
        id3.append(find1[i+2])
        id4.append(find1[i+3])
        id5.append(find1[i+4])

    #label 0
    for i in range(0,len(find0),5):
        # if i ==int(len(find0)/5)*5 and len(find0)%5!=0:#order
        #     t=len(find0)%5
        #     for j in range(t):
        #         id[4-j].append(find0[i+j])
        #     break
        if i ==int(len(find0)/5)*5 and len(find0)%5!=0:
            t=len(find0)%5
            for j in range(t):
                id[j].append(find0[i+j])
            break
        id5.append(find0[i])
        id4.append(find0[i+1])
        id3.append(find0[i+2])
        id2.append(find0[i+3])
        id1.append(find0[i+4])
    #shuffle
    for i in range(len(id)):
        random.shuffle(id[i])

    print()
    # with open("Ids_io216.txt","w") as f:
    #     for i in range(5):
    #         for j in range(len(id[i])):
    #             ten=str(id[i][j])
    #             f.write(ten+'\n')
    return id

























