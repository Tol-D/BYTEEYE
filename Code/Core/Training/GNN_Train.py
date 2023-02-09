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
import sys

sys.path.append("..")
from ..Models import GCN
from ..Models import GAT
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
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0165,  # 0.02 #0.005/sqrt(2)  #gn 0.04 0.02
                    help='Initial learning rate.')
parser.add_argument('--lr_decay_steps', type=str, default='1,3', help='learning rate')  # new
parser.add_argument('--weight_decay', type=float, default=0,  # 1e-4,#5e-4
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.21,  # 0.5 #0.2
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')  # 0.2
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--cfg', type=int, help='only build cfg')
parser.add_argument('--training', type=int, help='train the model and detect vulnerabilities')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# GPU
gpu=[0]
if torch.cuda.is_available():
    torch.cuda.set_device("cuda:{}".format(gpu[0]))
global rawadjs, rawfeatures,rawlabels,numbers,idx_train,idx_test,id
def preparation():
    global rawadjs, rawfeatures, rawlabels, numbers, idx_train, idx_test, id
    # Load data
    rawadjs, rawfeatures, numbers = dl2.load_data()#feature extraction important
    np.load.__defaults__ = (None, True, True, 'ASCII')  # for higher numpy version
    print("new scp reentrancy")
    with open('data/internal_result/adjs_scpre.npy', 'rb') as f:
        rawadjs = np.load(f)
    # rawadjs = torch.tensor(rawadjs,dtype=torch.float32)#when we turn ndarray into tensor, the accuracy of values will decrease
    with open('data/internal_result/features_scpre.npy', 'rb') as f:
        rawfeatures = np.load(f)
    # rawfeatures = torch.tensor(rawfeatures,dtype=torch.float32)
    with open('data/internal_result/numbers_scpre.npy', 'rb') as f:
        numbers = np.load(f)
    np.load.__defaults__ = (None, False, True, 'ASCII')
    numbers = numbers.tolist()
    rawlabels, idx_train, idx_valid, idx_test = dl2.load_datasets()

    id = dl2.kfolds(rawlabels)  # for 5-folds


# read ids_x.txt
def readid():
    global id
    with open("Ids_541.txt", "r") as f:
        idnew = f.readlines()
    cc = 0
    for i in range(len(id)):
        for j in range(len(id[i])):
            id[i][j] = int(idnew[cc].split('\n')[0])
            cc = cc + 1
    print()


record_control = []
record_trainm = []
record_total=[]


def padding(adjs, features, max_m):
    X, Y = [], []
    for i in range(len(features)):
        feature = np.pad(features[i], ((0, max_m - adjs[i].shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
        X.append(torch.Tensor(feature))
    for i in range(len(adjs)):
        adj = np.pad(adjs[i], ((0, max_m - adjs[i].shape[0]), (0, max_m - adjs[i].shape[0])), 'constant',
                     constant_values=(0, 0))
        Y.append(torch.Tensor(adj))
    features = torch.stack(X)
    adjs = torch.stack(Y)
    return adjs, features


# To compress the dataset, we design the function batch_padding
def batch_padding(fe, ad, queue):
    size = args.batch_size
    # deal with train dataset
    t = int(idx_train / size)
    for i in range(t):
        # compute the max value of a batch
        maxn = numbers[queue[i * size]]
        for j in range(size):
            if maxn < numbers[queue[i * size + j]]:
                maxn = numbers[queue[i * size + j]]
        # start padding
        tad, tfe = padding(ad[i * size:(i * size + size)], fe[i * size:(i * size + size)], maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
    if idx_train % size != 0:
        maxn = numbers[t * size]
        for i in range(t * size, idx_train):
            if maxn < numbers[queue[i]]:
                maxn = numbers[queue[i]]
        # start padding
        tad, tfe = padding(ad[(t * size):idx_train], fe[(t * size):idx_train], maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
        print("")
    # deal with test dataset
    t = int((idx_test - idx_train) / size)
    for i in range(t):
        # compute the max value of a batch
        maxn = numbers[queue[idx_train + i * size]]
        for j in range(size):
            if maxn < numbers[queue[idx_train + i * size + j]]:
                maxn = numbers[queue[idx_train + i * size + j]]
        # start padding
        tad, tfe = padding(ad[idx_train + i * size:(idx_train + i * size + size)],
                           fe[idx_train + i * size:(idx_train + i * size + size)], maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
    if (idx_test - idx_train) % size != 0:
        maxn = numbers[idx_train + t * size]
        for i in range(idx_train + t * size, idx_test):
            if maxn < numbers[queue[i]]:
                maxn = numbers[queue[i]]
        # start padding
        tad, tfe = padding(ad[(idx_train + t * size):idx_test], fe[(idx_train + t * size):idx_test], maxn)
        batch_features.append(tfe)
        batch_adjs.append(tad)
        print("")


def train_decided(K):
    ad = []
    fe = []
    la = []
    global adjs, features, labels
    global batch_features, batch_adjs
    batch_features = []
    batch_adjs = []
    queue = []
    for i in range(len(id)):
        if i != K:  # distribute train dataset
            for j in range(len(id[i])):
                ad.append(rawadjs[id[i][j]])
                fe.append(rawfeatures[id[i][j]])
                la.append(rawlabels[id[i][j]])
                queue.append(id[i][j])
    # change idx_train decided by dataset
    global idx_train
    idx_train = len(queue)
    for j in range(len(id[K])):
        ad.append(rawadjs[id[K][j]])
        fe.append(rawfeatures[id[K][j]])
        la.append(rawlabels[id[K][j]])
        queue.append(id[K][j])
    batch_padding(fe, ad, queue)
    # features=torch.stack(fe)
    # adjs=torch.stack(ad)
    labels = la
    # store variables
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
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if TP + FP == 0:  # Avoid division by 0
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:  # Avoid division by 0
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:  # Avoid division by 0
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    correct = preds.eq(label_a).double()
    correct = correct.sum()
    print("Temp results:",
          "accuracy= {:.4f}".format(accuracy),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "f1= {:.4f}".format(f1))
    record_control.append(("Temp results:", "accuracy= {:.4f}".format(accuracy), "precision= {:.4f}".format(precision),
                           "recall= {:.4f}".format(recall), "f1= {:.4f}".format(f1)))
    return preds, correct / len(label_a)


def accuracy_test(output_a, label_a, K):
    # preds = output_a.max(1)[1].type_as(label_a)
    preds = output_a
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
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if TP + FP == 0:  # Avoid division by 0
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:  # Avoid division by 0
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:  # Avoid division by 0
        f1 = 0
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
    wrongfn = []
    wrongfp = []
    for i in range(len(label_a)):
        if label_a[i].item() == 1 and output_a[i].item() == 0:
            wrongfn.append(id[K][i])
        if label_a[i].item() == 0 and output_a[i].item() == 1:
            wrongfp.append(id[K][i])
    record_control.append("FN ID is")
    record_control.append(wrongfn)
    record_control.append("FP ID is")
    record_control.append(wrongfp)
    record_control.append(("TP :{}".format(TP), "FN :{}".format(FN), "FP :{}".format(FP), "TN :{}".format(TN)))

    record_total.append([TP,FN,FP,TN])

    return preds, correct / len(label_a)


def main_task(K):
    args.model='gat'
    # Model and optimizer
    if args.model == 'gcn':
        model = GCN.GCN(  # nfeat=features.shape[1],
            nfeat=9,
            nhid=5,
            nclass=2,
            # nclass=labels.max().item() + 1,
            dropout=args.dropout)
    elif args.model == 'gat':
        model = GAT.GAT(nfeat=9,
                        nhid=5,
                        nclass=2,
                        dropout=args.dropout,
                        nheads=args.nb_heads,
                        alpha=args.alpha)
        if torch.cuda.is_available():
            model.cuda()
    # optimizer = optim.Adam(model.parameters(),
    #                      lr=args.lr, weight_decay=args.weight_decay)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)  # dynamic adjustment lr
    # weights = torch.tensor([1., 2.], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn=torch.nn.NLLLoss()
    # loss_fn= Focal_Loss.focal_loss(alpha=0.29, gamma=2, num_classes=2)
    if torch.cuda.is_available():
        loss_fn.cuda()

    record_loss = []
    record_acc = []
    record_validloss = []
    record_5folds = []

    def train(epoch):
        # scheduler.step()
        model.train()
        Prediction = []
        t = time.time()
        print('Epoch:', epoch + 1, 'lr:', scheduler.get_last_lr())
        record_control.append(['Epoch:', epoch + 1, 'lr:', scheduler.get_last_lr()])
        for index in range(int(idx_train / args.batch_size)):
            p = args.batch_size * index
            optimizer.zero_grad()
            temp_f = batch_features[index]
            temp_a = batch_adjs[index]
            temp_l = torch.tensor(labels[p:p + args.batch_size])
            if torch.cuda.is_available():
                temp_f = temp_f.cuda()
                temp_a = temp_a.cuda()
                temp_l = temp_l.cuda()
            output = model(temp_f, temp_a)

            # weights = torch.tensor([2., 1.], dtype=torch.float)
            # loss_f_none_w = torch.nn.NLLLoss(weight=weights, reduction='none')
            # loss_train =loss_f_none_w (output, torch.tensor(labels[p:p+args.batch_size]))
            # loss_train = F.nll_loss(output, torch.tensor(labels[p:p+args.batch_size]))
            # cross_entropy loss
            loss_train = loss_fn(output, temp_l)
            record_loss.append(loss_train.item())
            # focal loss
            # focal_loss_train=Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes = 2)
            # loss_train=focal_loss_train(output, torch.tensor(labels[p:p+args.batch_size]))

            preds, acc_train = accuracy(output, temp_l)
            Prediction.append(preds)
            # acc_train = accuracy(F.log_softmax(output, dim=1), labels[i])

            loss_train.backward()
            # new_loss.backward()
            optimizer.step()
        # consider that idx_train cannot be divisible by args.batch_size
        if int(idx_train / args.batch_size) * args.batch_size != idx_train:
            p = int(idx_train / args.batch_size) * args.batch_size
            optimizer.zero_grad()
            temp_f = batch_features[int(idx_train / args.batch_size)]
            temp_a = batch_adjs[int(idx_train / args.batch_size)]
            temp_l = torch.tensor(labels[p:idx_train])
            if torch.cuda.is_available():
                temp_f = temp_f.cuda()
                temp_a = temp_a.cuda()
                temp_l = temp_l.cuda()
            output = model(temp_f, temp_a)
            if temp_l.shape[0] == 1:
                output = output.reshape(1, 2)

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

            preds, acc_train = accuracy(output, temp_l)
            Prediction.append(preds)
            # acc_train = accuracy(F.log_softmax(output, dim=1), labels[i])

            loss_train.backward()
            # new_loss.backward()
            optimizer.step()
        scheduler.step()

        # compute overall acc_train
        # Prediction = torch.cat(Prediction).numpy().tolist()
        temp_l = torch.tensor(labels[0:idx_train])
        if torch.cuda.is_available():
            temp_l = temp_l.cuda()
        Prediction = torch.cat(Prediction)
        correct = Prediction.eq(temp_l).double()
        correct = correct.sum()

        # output = model(features[0:idx_train], adjs[0:idx_train])
        # loss_train2 = F.nll_loss(output, torch.tensor(labels[0:idx_train]))
        # focal_loss_train = Focal_Loss.focal_loss(alpha=0.28, gamma=2, num_classes=2)
        # loss_train2 = focal_loss_train(output, torch.tensor(labels[0:idx_train]))
        # loss_train2 = F.cross_entropy(output, torch.tensor(labels[0:idx_train]))

        # record_loss.append(loss_train2.item())
        record_acc.append((correct / len(labels[0:idx_train])).item())

        # print training message

        print('Epoch: {:04d}'.format(epoch + 1),
              # 'loss_train: {:.4f}'.format(loss_train2.item()),
              'overall acc_train: {:.4f}'.format(correct / len(labels[0:idx_train])),
              'time: {:.4f}s'.format(time.time() - t),
              'lr:{:.16f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        record_control.append(['Epoch: {:04d}'.format(epoch + 1),
                               # 'loss_train: {:.4f}'.format(loss_train2.item()),
                               'overall acc_train: {:.4f}'.format(correct / len(labels[0:idx_train])),
                               'time: {:.4f}s'.format(time.time() - t),
                               'lr:{:.16f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])])
        # optimizer.lr.numpy().item())
        print("------------------------------------------------------------")

        # save the data from the last 20 epochs
        tepoch = epoch + 1
        if tepoch > 200 and tepoch % 10 == 0:
            t = int(tepoch / 10) - 11
            torch.save(model.state_dict(), 'Core/Training/Training data/KSB{}_{}.pt'.format(K, t))
            record_trainm.append(
                ['Training message overall acc_train: {:.4f}'.format(correct / len(labels[0:idx_train]))])

    def test():
        model.eval()
        Prediction = []
        for index in range(int((idx_test - idx_train) / args.batch_size)):
            p = idx_train + args.batch_size * index
            if idx_train % args.batch_size != 0:
                t = int(idx_train / args.batch_size) + 1
            else:
                t = int(idx_train / args.batch_size)
            temp_f = batch_features[t + index]
            temp_a = batch_adjs[t + index]
            temp_l = torch.tensor(labels[p:p + args.batch_size])
            if torch.cuda.is_available():
                temp_f = temp_f.cuda()
                temp_a = temp_a.cuda()
                temp_l = temp_l.cuda()
            output = model(temp_f, temp_a)
            # loss_test = loss_fn(output, temp_l)
            preds, acc_train = accuracy(output, temp_l)
            Prediction.append(preds)
        # consider that idx_train cannot be divisible by args.batch_size
        if int((idx_test - idx_train) / args.batch_size) * args.batch_size != (idx_test - idx_train):
            p = idx_train + int((idx_test - idx_train) / args.batch_size) * args.batch_size
            temp_f = batch_features[-1]
            temp_a = batch_adjs[-1]
            temp_l = torch.tensor(labels[p:idx_test])
            if torch.cuda.is_available():
                temp_f = temp_f.cuda()
                temp_a = temp_a.cuda()
                temp_l = temp_l.cuda()
            output = model(temp_f, temp_a)
            if temp_l.shape[0] == 1:
                output = output.reshape(1, 2)
            # loss_test = loss_fn(output,temp_l)
            preds, acc_train = accuracy(output, temp_l)
            Prediction.append(preds)
        Prediction = torch.cat(Prediction)
        temp_l = torch.tensor(labels[idx_train:idx_test])
        if torch.cuda.is_available():
            temp_l = temp_l.cuda()
        preds, acc_test = accuracy_test(Prediction, temp_l, K)

        print("Test set results:",
              # "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        record_control.append(["Test set results:",
                               # "loss= {:.4f}".format(loss_test.item()),
                               "accuracy= {:.4f}".format(acc_test.item())])
        print(preds)

    def visualization():
        plt.figure(figsize=(10, 10))
        x = []
        for i in range(args.epochs):
            x.append(i)
        # plt.plot(x,record_loss,'b*--',alpha=0.5, linewidth=1,label='train_loss')
        # plt.plot(x, record_acc, 'rs--', alpha=0.5, linewidth=1, label='train_acc')
        # plt.plot(x, record_validloss, 'go--', alpha=0.5, linewidth=1, label='valid_loss')
        t = int(len(record_loss) / len(record_acc))
        trecord_loss = []
        for i in range(len(record_acc)):
            tem = 0
            for j in range(t):
                tem = tem + record_loss[i * t + j]
            trecord_loss.append(tem / t)
        plt.plot(x, trecord_loss, 'b-', alpha=0.5, linewidth=1, label='train_loss')
        plt.plot(x, record_acc, 'r-', alpha=0.5, linewidth=1, label='train_acc')
        # plt.plot(x, record_validloss, 'g-', alpha=0.5, linewidth=1, label='valid_loss')
        plt.legend()  # show labels
        plt.xlabel('epochs')
        plt.ylabel('value')
        # plt.scatter(x,record_loss)
        plt.show()

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test()
    # visualization()
def print_statistic():
    TP,FN,FP,TN=0,0,0,0
    for i in range(len(record_total)):
        TP=TP+record_total[i][0]
        FN=FN+record_total[i][1]
        FP=FP+record_total[i][2]
        TN=TN+record_total[i][3]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if TP + FP == 0:  # Avoid division by 0
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:  # Avoid division by 0
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:  # Avoid division by 0
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    print("total results: ")
    print("accuracy is {}".format(accuracy))
    print("precision is {}".format(precision))
    print("recall is {}".format(recall))
    print("f1 is {}".format(f1))

def main_training():
    #preparation
    preparation()
    readid()
    # cross validiations  TODO
    for K in range(len(id)):
        print("Now the {:04d}th is test dataset".format(K))
        record_control.append(["Now the {:04d}th is test dataset".format(K)])
        train_decided(K)  # K decides that the Kth become the test dataset.
        main_task(K)

    print("I'm free!")

    #Satistics
    print_statistic()
    #save results in the file
    file = open('Core/Training/Training data/training_result.txt', 'w')
    for i in range(len(record_control)):
        file.write(str(record_control[i]) + '\n')
    file.close()




