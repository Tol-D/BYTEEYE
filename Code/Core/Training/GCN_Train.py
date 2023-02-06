from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#from torchinfo import summary
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append("..")
from ..Models import GCN
from ..Models import GAT
from . import Data_Load as dl
from . import Data_Load2 as dl2
from . import Focal_Loss
'''from pygcn.utils import load_data, accuracy
from pygcn.models import GCN'''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,#0.02 #0.005/sqrt(2)  #gn 0.04 0.02
                    help='Initial learning rate.')
parser.add_argument('--lr_decay_steps', type=str, default='1,3', help='learning rate')#new
parser.add_argument('--weight_decay', type=float, default=0,#1e-4,#5e-4
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,#0.5 #0.2
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=8,help='batch_size')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')#0.2
parser.add_argument('--nb_heads', type=int, default=6, help='Number of head attentions.')
parser.add_argument('--model', type=str, default='gcn',choices=['gcn', 'gat'])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#GPUbbbb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
rawadjs, rawfeatures,numbers = dl2.load_data()
np.load.__defaults__=(None, True, True, 'ASCII')#for higher numpy version
print("new scp reentrancy")
with open('data/internal_result/adjs_scpre.npy', 'rb') as f:
    rawadjs = np.load(f)
#rawadjs = torch.tensor(rawadjs,dtype=torch.float32)#when we turn ndarray into tensor, the accuracy of values will decrease
with open('data/internal_result/features_scpre.npy', 'rb') as f:
    rawfeatures = np.load(f)
#rawfeatures = torch.tensor(rawfeatures,dtype=torch.float32)
with open('data/internal_result/numbers_scpre.npy', 'rb') as f:
    numbers = np.load(f)
np.load.__defaults__=(None, False, True, 'ASCII')
numbers=numbers.tolist()
rawlabels,idx_train,idx_valid,idx_test=dl2.load_datasets()
#adjs, features,labels=dl2.re_balance(adjs, features,labels,idx_train,idx_valid,idx_test)
id=dl2.kfolds(rawlabels) #for 5-folds
#read ids_x.txt
def readid():
    with open("Ids_i541.txt","r") as f:
        idnew=f.readlines()
    cc=0
    for i in range(len(id)):
        for j in range(len(id[i])):
            id[i][j]=int(idnew[cc].split('\n')[0])
            cc=cc+1
    print()
readid()

record_control=[]
record_trainm=[]

def padding(adjs,features,max_m):
    X,Y=[],[]
    for i in range(len(features)):
        feature=np.pad(features[i],((0,max_m-adjs[i].shape[0]),(0,0)),'constant',constant_values=(0,0))
        X.append(torch.Tensor(feature))
    for i in range(len(adjs)):
        adj=np.pad(adjs[i],((0,max_m-adjs[i].shape[0]),(0,max_m-adjs[i].shape[0])),'constant',constant_values=(0,0))
        Y.append(torch.Tensor(adj))
    features=torch.stack(X)
    adjs=torch.stack(Y)
    return adjs,features

#To compress the dataset, we design the function batch_padding
def batch_padding(fe,ad,queue):
    size=args.batch_size
    #deal with train dataset
    t = int(idx_train / size)
    for i in range(t):
        #compute the max value of a batch
        maxn=numbers[queue[i*size]]
        for j in range(size):
            if maxn<numbers[queue[i*size+j]]:
                maxn=numbers[queue[i*size+j]]
        #start padding
        tad,tfe=padding(ad[i*size:(i*size+size)],fe[i*size:(i*size+size)],maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
    if idx_train %size!=0:
        maxn = numbers[t * size]
        for i in range(t*size,idx_train):
            if maxn < numbers[queue[i]]:
                maxn = numbers[queue[i]]
        # start padding
        tad, tfe = padding(ad[(t*size):idx_train], fe[(t*size):idx_train], maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
        print("")
    #deal with test dataset
    t = int((idx_test-idx_train) / size)
    for i in range(t):
        # compute the max value of a batch
        maxn = numbers[queue[idx_train+i * size]]
        for j in range(size):
            if maxn < numbers[queue[idx_train+i * size + j]]:
                maxn = numbers[queue[idx_train+i * size + j]]
        # start padding
        tad, tfe = padding(ad[idx_train+i * size:(idx_train+i * size + size)], fe[idx_train+i * size:(idx_train+i * size + size)], maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
    if (idx_test-idx_train)  % size != 0:
        maxn = numbers[idx_train+t * size]
        for i in range(idx_train+t * size, idx_test):
            if maxn < numbers[queue[i]]:
                maxn = numbers[queue[i]]
        # start padding
        tad, tfe = padding(ad[(idx_train+t * size):idx_test], fe[(idx_train+t * size):idx_test], maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
        print("")


def train_decided(K):
    ad=[]
    fe=[]
    la=[]
    global adjs,features,labels
    global batch_features,batch_adjs
    batch_features=[]
    batch_adjs=[]
    queue=[]
    for i in range(len(id)):
        if i!=K:#distribute train dataset
            for j in range(len(id[i])):
                ad.append(rawadjs[id[i][j]])
                fe.append(rawfeatures[id[i][j]])
                la.append(rawlabels[id[i][j]])
                queue.append(id[i][j])
    # change idx_train decided by dataset
    global  idx_train
    idx_train=len(queue)
    for j in range(len(id[K])):
        ad.append(rawadjs[id[K][j]])
        fe.append(rawfeatures[id[K][j]])
        la.append(rawlabels[id[K][j]])
        queue.append(id[K][j])
    batch_padding(fe,ad,queue)
    #features=torch.stack(fe)
    #adjs=torch.stack(ad)
    labels=la
    #store variables
    print("")

def accuracy(output_a, label_a):
    preds = output_a.max(1)[1].type_as(label_a)
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(preds)):
        if preds[i] == 0 and label_a[i] == 0:
            TN = TN + 1
        elif preds[i] == 1 and label_a[i] == 1:
            TP = TP + 1
        elif preds[i] == 1 and label_a[i] == 0:
            FP = FP + 1
        elif preds[i] == 0 and label_a[i] == 1:
            FN = FN + 1
    accuracy=(TP + TN) / (TP + TN + FP + FN)
    if TP+FP==0:#Avoid division by 0
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP+FN==0:#Avoid division by 0
        recall=0
    else:
        recall = TP / (TP + FN)
    if precision+recall==0:#Avoid division by 0
        f1=0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    correct = preds.eq(label_a).double()
    correct = correct.sum()
    print("Temp results:",
          "accuracy= {:.4f}".format(accuracy),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "f1= {:.4f}".format(f1))
    record_control.append(("Temp results:","accuracy= {:.4f}".format(accuracy),"precision= {:.4f}".format(precision),"recall= {:.4f}".format(recall),"f1= {:.4f}".format(f1)))
    return preds,correct / len(label_a)

def accuracy_test(output_a, label_a,K):
    #preds = output_a.max(1)[1].type_as(label_a)
    preds=output_a
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(preds)):
        if preds[i] == 0 and label_a[i] == 0:
            TN = TN + 1
        elif preds[i] == 1 and label_a[i] == 1:
            TP = TP + 1
        elif preds[i] == 1 and label_a[i] == 0:
            FP = FP + 1
        elif preds[i] == 0 and label_a[i] == 1:
            FN = FN + 1
    accuracy=(TP + TN) / (TP + TN + FP + FN)
    if TP+FP==0:#Avoid division by 0
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP+FN==0:#Avoid division by 0
        recall=0
    else:
        recall = TP / (TP + FN)
    if precision+recall==0:#Avoid division by 0
        f1=0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    correct = preds.eq(label_a).double()
    correct = correct.sum()
    print("Test results:",
          "accuracy= {:.4f}".format(accuracy),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "f1= {:.4f}".format(f1))
    record_control.append(("Test results:", "accuracy= {:.4f}".format(accuracy),
                           "precision= {:.4f}".format(precision), "recall= {:.4f}".format(recall),
                           "f1= {:.4f}".format(f1)))
    wrongfn=[]
    wrongfp=[]
    for i in range(len(label_a)):
        if label_a[i].item()==1 and output_a[i].item()==0:
            wrongfn.append(id[K][i])
        if label_a[i].item()==0 and output_a[i].item()==1:
            wrongfp.append(id[K][i])
    record_control.append("FN ID is")
    record_control.append(wrongfn)
    record_control.append("FP ID is")
    record_control.append(wrongfp)
    record_control.append(("TP :{}".format(TP),"FN :{}".format(FN),"FP :{}".format(FP),"TN :{}".format(TN)))
    #record_5folds.append([accuracy,precision,recall,f1])
    return preds,correct / len(label_a)

def main_task(K):
    #args.model='gat'
    # Model and optimizer
    if args.model=='gcn':
        model = GCN.GCN(#nfeat=features.shape[1],
                        nfeat=9,
                        nhid=5,
                        nclass=2,
                        # nclass=labels.max().item() + 1,
                        dropout=args.dropout)
    elif args.model=='gat':
        model = GAT.GAT(nfeat=9,
                    nhid=5,
                    nclass=2,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
        model.to(device)
    #optimizer = optim.Adam(model.parameters(),
     #                      lr=args.lr, weight_decay=args.weight_decay)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.999), weight_decay=args.weight_decay)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)  # dynamic adjustment lr
    weights = torch.tensor([1., 2.], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    #loss_fn=torch.nn.NLLLoss()
    #loss_fn= Focal_Loss.focal_loss(alpha=0.29, gamma=2, num_classes=2)
    loss_fn.to(device)

    #check the structure of model
    #summary(your_model, input_size=(channels, H, W))
    #summary(model, input_size=(32, 40, 40, 40))
    #print(model)
    record_loss=[]
    record_acc=[]
    record_validloss=[]
    record_5folds=[]

    def train(epoch):
        #scheduler.step()
        model.train()
        Prediction=[]
        t = time.time()
        print('Epoch:',epoch+1,'lr:',scheduler.get_last_lr())
        record_control.append(['Epoch:',epoch+1,'lr:',scheduler.get_last_lr()])
        for index in range(int(idx_train/args.batch_size)):
            p=args.batch_size*index
            optimizer.zero_grad()
            temp_f = batch_features[index]
            temp_a = batch_adjs[index]
            temp_l = torch.tensor(labels[p:p + args.batch_size])
            temp_f = temp_f.to(device)
            temp_a = temp_a.to(device)
            temp_l = temp_l.to(device)
            output = model(temp_f, temp_a)

            #weights = torch.tensor([2., 1.], dtype=torch.float)
            #loss_f_none_w = torch.nn.NLLLoss(weight=weights, reduction='none')
            #loss_train =loss_f_none_w (output, torch.tensor(labels[p:p+args.batch_size]))
            #loss_train = F.nll_loss(output, torch.tensor(labels[p:p+args.batch_size]))
            #cross_entropy loss
            loss_train = loss_fn(output, temp_l)
            record_loss.append(loss_train.item())
            #focal loss
            # focal_loss_train=Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes = 2)
            # loss_train=focal_loss_train(output, torch.tensor(labels[p:p+args.batch_size]))

            preds,acc_train = accuracy(output,temp_l)
            Prediction.append(preds)
            #acc_train = accuracy(F.log_softmax(output, dim=1), labels[i])

            loss_train.backward()
            #new_loss.backward()
            optimizer.step()
        #consider that idx_train cannot be divisible by args.batch_size
        if int(idx_train/args.batch_size)*args.batch_size!=idx_train:
            p= int(idx_train/args.batch_size)* args.batch_size
            optimizer.zero_grad()
            temp_f =batch_features[int(idx_train/args.batch_size)]
            temp_a = batch_adjs[int(idx_train/args.batch_size)]
            temp_l = torch.tensor(labels[p:idx_train])
            temp_f = temp_f.to(device)
            temp_a = temp_a.to(device)
            temp_l = temp_l.to(device)
            output = model(temp_f,temp_a)
            if temp_l.shape[0]==1:
                output=output.reshape(1,2)

            # weights = torch.tensor([2., 1.], dtype=torch.float)
            # loss_f_none_w = torch.nn.NLLLoss(weight=weights, reduction='none')
            # loss_train =loss_f_none_w (output, torch.tensor(labels[p:p+args.batch_size]))
            # loss_train = F.nll_loss(output, torch.tensor(labels[p:idx_train]))
            # cross_entropy loss
            loss_train = loss_fn(output, temp_l)
            record_loss.append(loss_train.item())
            # focal loss
            # focal_loss_train = Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes=2)
            # loss_train = focal_loss_train(output, torch.tensor(labels[p:idx_train]))

            preds, acc_train = accuracy(output,temp_l)
            Prediction.append(preds)
            # acc_train = accuracy(F.log_softmax(output, dim=1), labels[i])

            loss_train.backward()
            # new_loss.backward()
            optimizer.step()
        scheduler.step()

        #compute overall acc_train
        #Prediction = torch.cat(Prediction).numpy().tolist()
        temp_l = torch.tensor(labels[0:idx_train])
        temp_l = temp_l.to(device)
        Prediction = torch.cat(Prediction)
        correct = Prediction.eq(temp_l).double()
        correct = correct.sum()

        #output = model(features[0:idx_train], adjs[0:idx_train])
        #loss_train2 = F.nll_loss(output, torch.tensor(labels[0:idx_train]))
        # focal_loss_train = Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes=2)
        # loss_train2 = focal_loss_train(output, torch.tensor(labels[0:idx_train]))
        #loss_train2 = F.cross_entropy(output, torch.tensor(labels[0:idx_train]))

        #record_loss.append(loss_train2.item())
        record_acc.append((correct/len(labels[0:idx_train])).item())

        #print training message

        print('Epoch: {:04d}'.format(epoch+1),
              #'loss_train: {:.4f}'.format(loss_train2.item()),
              'overall acc_train: {:.4f}'.format(correct/len(labels[0:idx_train])),
              'time: {:.4f}s'.format(time.time() - t),
              'lr:{:.16f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        record_control.append(['Epoch: {:04d}'.format(epoch+1),
              #'loss_train: {:.4f}'.format(loss_train2.item()),
              'overall acc_train: {:.4f}'.format(correct/len(labels[0:idx_train])),
              'time: {:.4f}s'.format(time.time() - t),
              'lr:{:.16f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])])
        #optimizer.lr.numpy().item())
        print("------------------------------------------------------------")

        #save the data from the last 20 epochs
        tepoch=epoch+1
        if tepoch>200 and tepoch%10==0:
            t=int(tepoch/10)-11
            torch.save(model.state_dict(),'Core/Training/Training data/KSB{}_{}.pt'.format(K,t))
            record_trainm.append(['Training message overall acc_train: {:.4f}'.format(correct / len(labels[0:idx_train]))])

       #  #For validation: hyper-parameter
       #  output = model(features[idx_train:idx_valid], adjs[idx_train:idx_valid])
       #  # loss_valid = F.nll_loss(output, torch.tensor(labels[idx_train:idx_valid]))
       #  #loss_valid = F.cross_entropy(output, torch.tensor(labels[idx_train:idx_valid]))
       #  # focal loss
       #  focal_loss_train = Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes=2)
       #  loss_valid = focal_loss_train(output, torch.tensor(labels[idx_train:idx_valid]))
       # # preds, acc_valid = accuracy(output, torch.tensor(labels[idx_train:idx_valid]))
       #  record_validloss.append(loss_valid.item())

    def test():
        model.eval()
        Prediction = []
        for index in range(int((idx_test - idx_train) / args.batch_size)):
            p = idx_train + args.batch_size * index
            if idx_train%args.batch_size!=0:
                t=int(idx_train/ args.batch_size)+1
            else:
                t=int(idx_train/ args.batch_size)
            temp_f = batch_features[t+index]
            temp_a = batch_adjs[t+index]
            temp_l = torch.tensor(labels[p:p + args.batch_size])
            temp_f = temp_f.to(device)
            temp_a = temp_a.to(device)
            temp_l = temp_l.to(device)
            output = model(temp_f, temp_a)
            # loss_test = loss_fn(output, temp_l)
            preds, acc_train = accuracy(output, temp_l)
            Prediction.append(preds)
        # consider that idx_train cannot be divisible by args.batch_size
        if int((idx_test - idx_train) / args.batch_size) * args.batch_size != (idx_test - idx_train):
            p = idx_train+int((idx_test - idx_train) / args.batch_size) * args.batch_size
            temp_f = batch_features[-1]
            temp_a = batch_adjs[-1]
            temp_l = torch.tensor(labels[p:idx_test])
            temp_f = temp_f.to(device)
            temp_a = temp_a.to(device)
            temp_l = temp_l.to(device)
            output = model(temp_f, temp_a)
            if temp_l.shape[0]==1:
                output=output.reshape(1,2)
            # loss_test = loss_fn(output,temp_l)
            preds, acc_train = accuracy(output, temp_l)
            Prediction.append(preds)
        Prediction = torch.cat(Prediction)
        temp_l = torch.tensor(labels[idx_train:idx_test])
        temp_l = temp_l.to(device)
        preds, acc_test = accuracy_test(Prediction, temp_l,K)

        #output = model(features[idx_train:idx_test], adjs[idx_train:idx_test])
        # loss_test =  F.nll_loss(output, torch.tensor(labels[idx_valid:idx_test]))
        #loss_test =  F.cross_entropy(output, torch.tensor(labels[idx_train:idx_test]))
        #focal loss
        # focal_loss_train = Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes=2)
        # loss_test = focal_loss_train(output, torch.tensor(labels[idx_train:idx_test]))

        #acc_test = accuracy(F.log_softmax(output, dim=1), labels[i])
        #preds, acc_test = accuracy_test(output, torch.tensor(labels[idx_train:idx_test]))
        print("Test set results:",
               #"loss= {:.4f}".format(loss_test.item()),
               "accuracy= {:.4f}".format(acc_test.item()))
        record_control.append(["Test set results:",
               #"loss= {:.4f}".format(loss_test.item()),
               "accuracy= {:.4f}".format(acc_test.item())])
        print(preds)

    # def test():
    #     model.eval()
    #
    #     output = model(features[idx_valid:idx_test], adjs[idx_valid:idx_test])
    #     # loss_test =  F.nll_loss(output, torch.tensor(labels[idx_valid:idx_test]))
    #     #loss_test =  F.cross_entropy(output, torch.tensor(labels[idx_valid:idx_test]))
    #     #focal loss
    #     focal_loss_train = Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes=2)
    #     loss_test = focal_loss_train(output, torch.tensor(labels[idx_valid:idx_test]))
    #
    #     #acc_test = accuracy(F.log_softmax(output, dim=1), labels[i])
    #     preds, acc_test = accuracy(output, torch.tensor(labels[idx_valid:idx_test]))
    #     print("Test set results:",
    #            "loss= {:.4f}".format(loss_test.item()),
    #            "accuracy= {:.4f}".format(acc_test.item()))
    #     print(preds)

    def visualization():
        plt.figure(figsize=(10,10))
        x=[]
        for i in range(args.epochs):
            x.append(i)
        # plt.plot(x,record_loss,'b*--',alpha=0.5, linewidth=1,label='train_loss')
        # plt.plot(x, record_acc, 'rs--', alpha=0.5, linewidth=1, label='train_acc')
        # plt.plot(x, record_validloss, 'go--', alpha=0.5, linewidth=1, label='valid_loss')
        t=int(len(record_loss)/len(record_acc))
        trecord_loss=[]
        for i in range (len(record_acc)):
            tem=0
            for j in range(t):
                tem=tem+record_loss[i*t+j]
            trecord_loss.append(tem/t)
        plt.plot(x,trecord_loss,'b-',alpha=0.5, linewidth=1,label='train_loss')
        plt.plot(x, record_acc, 'r-', alpha=0.5, linewidth=1, label='train_acc')
        #plt.plot(x, record_validloss, 'g-', alpha=0.5, linewidth=1, label='valid_loss')
        plt.legend() #show labels
        plt.xlabel('epochs')
        plt.ylabel('value')
        #plt.scatter(x,record_loss)
        plt.show()


    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test()
    #visualization()

def find_best(K):
    model2 = GAT.GAT(nfeat=9,
                    nhid=5,
                    nclass=2,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    adjs = []
    features = []
    labels = []
    np.load.__defaults__ = (None, True, True, 'ASCII')  # for higher numpy version
    with open('/Core/Training/Training data/adj_KSB{}.npy'.format(K), 'rb') as f:
        a = np.load(f)
    for i in range(len(a)):
        adjs.append(a[i])
    with open('Core/Training/Training data/feature_KSB{}.npy'.format(K), 'rb') as f:
        b = np.load(f)
    for i in range(len(b)):
        features.append(b[i])
    with open('Core/Training/Training data/label_KSB{}.npy'.format(K), 'rb') as f:
        c = np.load(f)
    for i in range(len(c)):
        labels.append(c[i])
    np.load.__defaults__ = (None, False, True, 'ASCII')
    for i in range(30):
        tempi=i+10
        model2.load_state_dict(torch.load('Core/Training/Training data/KSB{}_{}.pt'.format(K, tempi)))
        print("test KSB{}_{}.pt".format(K, tempi))
        record_control.append(["test KSB{}_{}.pt".format(K, tempi)])
        # test dataset message
        Prediction=[]
        for j in range(len(features)):
            output = model2(features[j], adjs[j])
            preds, acc_test = accuracy(output,torch.Tensor(labels[j*args.batch_size:j*args.batch_size+args.batch_size]))
            Prediction.append(preds)
        Prediction = torch.cat(Prediction)
        preds, acc_test = accuracy_test(Prediction, torch.Tensor(labels))
    file = open('Core/Training/Training data/training_find_best{}.txt'.format(K), 'w')
    for i in range(len(record_control)):
        file.write(str(record_control[i]) + '\n')
    file.close()
    record_control.clear()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        # record_control.append(["Test set results:",
        #                        "loss= {:.4f}".format(loss_test.item()),
        #                        "accuracy= {:.4f}".format(acc_test.item())])


def main_training():
    #cross validiations  TODO
    for K in range(len(id)):
        print("Now the {:04d}th is test dataset".format(K))
        record_control.append(["Now the {:04d}th is test dataset".format(K)])
        train_decided(K)#K decides that the Kth become the test dataset.
        main_task(K)
 
        # find_best(K)

    #print(record_5folds)

    print("I'm free!")
    #visualization()

    file=open('Core/Training/Training data/trainingnew.txt','w')
    for i in range(len(record_control)):
        file.write(str(record_control[i])+'\n')
    file.close()

    


