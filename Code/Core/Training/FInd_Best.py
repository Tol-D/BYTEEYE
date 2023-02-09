import csv
import re
# def find_best():
#     model = GAT.GAT(nfeat=6,
#                     nhid=5,
#                     nclass=2,
#                     dropout=args.dropout,
#                     nheads=args.nb_heads,
#                     alpha=args.alpha)
#
#     for K in range(5):
#         adjs=[]
#         features=[]
#         labels=[]
#         np.load.__defaults__ = (None, True, True, 'ASCII')  # for higher numpy version
#         with open('/home/dell/yjn/ByteEye/Core/Training/Training data/adj_KSB{}.npy'.format(K), 'rb') as f:
#             a = np.load(f)
#         for i in range(len(a)):
#             adjs.append(a[i])
#         with open('/home/dell/yjn/ByteEye/Core/Training/Training data/feature_KSB{}.npy'.format(K), 'rb') as f:
#             b = np.load(f)
#         for i in range(len(b)):
#             features.append(b[i])
#         with open('/home/dell/yjn/ByteEye/Core/Training/Training data/label_KSB{}.npy'.format(K), 'rb') as f:
#             c = np.load(f)
#         for i in range(len(c)):
#             labels.append(c[i])
#         np.load.__defaults__ = (None, False, True, 'ASCII')
#         for i in range(40):
#             model.load_state_dict(torch.load('/home/dell/yjn/ByteEye/Core/Training/Training data/KSB{}_{}.pt'.format(K, i)))
#             print("test KSB{}_{}.pt".format(K, i))
#             record_control.append(["test KSB{}_{}.pt".format(K, i)])
#             # test dataset message
#             output = model(features[idx_train:idx_test], adjs[idx_train:idx_test])
#             loss_test = torch.nn.cross_entropy(output, torch.tensor(labels[idx_train:idx_test]))
#             preds, acc_test = accuracy_test(output, torch.tensor(labels[idx_train:idx_test]))
#             print("Test set results:",
#                   "loss= {:.4f}".format(loss_test.item()),
#                   "accuracy= {:.4f}".format(acc_test.item()))
#             record_control.append(["Test set results:",
#                                    "loss= {:.4f}".format(loss_test.item()),
#                                    "accuracy= {:.4f}".format(acc_test.item())])

f = open('/home/dell/yjn/ByteEye/Core/Training/Training data/training_analysis.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["Epoch", "TP", "FN","FP", "TN", "Accuracy", "Precision", "Recall","F1"])
def compute(Epoch,TP,FN,FP,TN):
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
    temp=[Epoch,TP,FN,FP,TN,accuracy,precision,recall,f1]
    return temp
for K in range(5):
    file = open('/home/dell/yjn/ByteEye/Core/Training/Training data/training_find_best{}.txt'.format(K), 'r')
    try:
        lines = file.readlines()
    finally:
        file.close()
    record=False
    Epoch=0
    for line in lines:
        if record and 'TP' in line:
            TP,FN,FP,TN=0,0,0,0
            t = re.split(':|,|\'', line)
            for i in range(len(t)):
                if 'TP' in t[i]:
                    TP=int(t[i+1])
                if 'FN' in t[i]:
                    FN=int(t[i+1])
                if 'FP' in t[i]:
                    FP=int(t[i+1])
                if 'TN' in t[i]:
                    TN=int(t[i+1])
            csv_writer.writerow(compute(Epoch,TP,FN,FP,TN))
            record=False
        if 'test KSB{}_'.format(K) in line and '9' in line:
            Epoch=200+100*int((int(re.split('_|\.',line)[1])+1)/10-1)
            record=True
    csv_writer.writerow([])
f.close()
