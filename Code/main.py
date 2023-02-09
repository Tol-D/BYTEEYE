import argparse
from Core.CFG import cfg_Construct
from Core.Training import GNN_Train

import time

def main():
    #get inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg',type=int, help='only build cfg')
    parser.add_argument('--training', type=int, help='train the model and detect vulnerabilities')

    args=parser.parse_args()

    print("main procedure")
    if args.cfg==None and args.training==None:
        start = time.time()
        cfg_Construct.main()
        end = time.time()
        print('total time is {}'.format(end - start))
        GNN_Train.main_training()
    elif args.cfg==1:
        # --build cfg
        start=time.time()
        cfg_Construct.main()
        end=time.time()
        print('total time is {}'.format(end-start))
    elif args.training==1:
        #--train and predict
        GNN_Train.main_training()

if __name__ == '__main__':
    main()

