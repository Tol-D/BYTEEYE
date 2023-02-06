from Core.CFG import cfg_Construct
# from Core.Training import GCN_Train

import time

def main():
    print("main procedure")
    #Task1: Data preprocess--generate Runtime Bytecode   [Module Input]
    #.......
    #Task2: Pure code generation    [Module Contracts]
    #......
    #Task3: --build cfg     [Module CFG] [Module Opcode]
    # To construct a secure analysis tool like ByteEye, we first need to construct an accurate cfg.
    start=time.time()
    cfg_Construct.main()
    end=time.time()
    print('total time is {}'.format(end-start))
    #Task4: --detect potential bugs     [Module Detection]
    # ......

    #Task5: --analyze experiment result     [Module Analysis]
    GCN_Train.main_training()

if __name__ == '__main__':
    main()

