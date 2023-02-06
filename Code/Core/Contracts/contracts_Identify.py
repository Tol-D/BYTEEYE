import csv
import re
#Smart contracts have different compile versions, sources and structure


#To deal with the contracts from test_contract.csv
def remove_metadata(bytecode,judge):
    if judge!="sbd":
        # for version 0.5.12  remove metadata
        version='v0.5.12'
        next=False
        bytecode_nometa=''
        try:
            #version 0.5.12
            sp = 'a265627a7a72315820[0-9a-f]{64}64736f6c6343[0-9a-f]{6}0032'
            stringb = ''.join(bytecode)
            test=re.split(sp,stringb)[1]
        except IndexError:
            next=True
        else:
            #belong to version 0.5.12
            bytecode_nometa = re.split(sp, stringb)[0]
        if next:
            try:
                #version 0.4.?  17?25?
                sp2 = 'a165627a7a72305820[0-9a-f]{64}0029'
                stringb = ''.join(bytecode)
                test = re.split(sp2, stringb)[1]
            except IndexError:
                print("new version!!!!ATTENTATION")
            else:
                bytecode_nometa = re.split(sp2, stringb)[0]
                version='v0.4.17'
    else:
        bytecode_nometa=bytecode
        if bytecode_nometa[-4:]=="0033":#0.6.10
            newv=bytecode_nometa[:-108]
            version="v0.6.10"
            return newv, version
        while bytecode_nometa[-4:]=="0029" or bytecode_nometa[-4:]=="0032":
            bytecode_nometa=bytecode_nometa[:-86]
        version="sbd"
        print("")
    return bytecode_nometa,version

