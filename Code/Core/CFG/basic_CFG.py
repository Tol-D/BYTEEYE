import sys
sys.path.append("..")
#from Opcode import opcode_Execute as oe
from ..Opcode import opcode_Execute as oe  #RUN main.py


# turn a str to a int
def str2int(str):
    intermediate = bytes(str, 'UTF-8')
    str_int = int.from_bytes(intermediate, 'big')
    end = 0
    if (str_int > 47 and str_int < 58):
        end = str_int - 48
    elif (str_int > 96 and str_int < 103):
        end = str_int - 87
    return end
# For test: count the block with the pattern of easy jump
def count_easyJump(edges):
    easyJump_block=[]
    for edge in edges:
        if edge[0] not in easyJump_block:
            easyJump_block.append(edge[0])
    return len(easyJump_block)


class basicCFG:
    def __init__(self,nodes,num):
        self.nodes=nodes
        self.edges=[]
        self.block_index=[0]
        self.global_order=[]#Record the execution orders(global)
        self.execute_order=[]#Record the execution order
        self.execute_routes=[]#Record execution routes
        self.executed=[]
        self.T=0
        self.num=num #Record the Xth smart contract
        self.wrongInfo=[]

    #Execute the instructions and compute the jump target
    def execute_JUMPtarget(self,bytecode,source):
        jump_target,only_AMode = oe.op_execution(bytecode, 0)
        if jump_target ==[]:# Unresolved jump Todo
                return source,'un'
        else:
            temp=0
            for i in range(len(jump_target[0])):
                pow_result=pow(16,len(jump_target[0])-i-1)
                temp=temp+pow_result*str2int(jump_target[0][i])
            #check whether temp belongs to block index or not
            if temp in self.block_index:
                edge=[]
                edge.append(source)
                edge.append(temp)
                if only_AMode==True:
                    edge.append("CA")
                else:
                    edge.append("CB")
                #avoid duplication
                if edge not in self.edges:
                    self.edges.append(edge)
            else:
                # if jump_target[0]=='20':
                #     pass
                # else:
                record_edge=[]
                record_edge.append(self.num)
                record_edge.append(jump_target[0])
                record_edge.append(source)
                record_edge.append(temp)
                record_edge.append(bytecode)
                self.wrongInfo.append(record_edge)
                print("wrong edge ")

            return source,only_AMode


    def check_Pedge(self,index):
        block_set = []
        for p_index in range(len(self.edges)):
            if (self.block_index[index] == self.edges[p_index][1]):
                temp = self.edges[p_index][0]
                block_set.append(self.block_index.index(temp))
        return block_set

    #recursion
    #At first, we don't check other stacks' depth
    def check_stk(self,index,str_stk):
        block_set=self.check_Pedge(index)
        #avoid recursion
        for i in self.execute_order:
            for j in range(len(block_set)-1,-1,-1):
                if block_set[j]==i[1]:
                    del block_set[j]
        # if(str_stk<=0):
        #new change 2023/1/23
        if (str_stk <= 0 and self.nodes[index].startStack<=0):
            print("current stop station is "+str(index))
            return str_stk
        else:
            for i in block_set:
                self.T=self.T+1
                if self.T==15:# define the largest recursion depth
                    return str_stk
                self.check_stk(i,str_stk-self.nodes[i].addStack)
                self.T=self.T-1
                print(str(i)+" ->" +str(index))
                self.execute_order.append([i,index])

    # #recursion
    # #At first, we don't check other stacks' depth
    # def check_stk(self,index,str_stk):
    #     block_set=self.check_Pedge(index)
    #     if(str_stk<=0):
    #         print("current stop station is "+str(index))
    #         return str_stk
    #     else:
    #         for i in block_set:
    #             self.check_stk(i,str_stk-self.nodes[i].addStack)
    #             print(str(i)+" ->" +str(index))
    #             self.execute_order.append([i,index])
    def retry(self,route,wrongadd,p):
        self.check_stk(route[wrongadd][1], self.nodes[route[wrongadd][1]].startStack)
        for i in range(len(self.execute_order)):
            if self.execute_order[i]==self.global_order[-1][0]:
                for j in range(i-1,-1,-1):
                    self.global_order[-1].insert(0,self.execute_order[j])
                break
        self.retryx=False
        self.generate_routes(p)
        print()
    def generate_routes(self,p):
        # split the different routes in global_order
        whole_routes=[]
        for i in range(p,len(self.global_order)):
            dest=self.global_order[i][-1][1]
            temp=[self.global_order[i][-1]]
            final_list=[self.global_order[i][-1]]
            for j in range(len(self.global_order[i])-1,-1,-1):
                if (self.global_order[i][j][0] == self.global_order[i][j -1][1]):
                    temp.append(self.global_order[i][j-1])
                    if final_list[-1]!='x':
                        final_list.append(self.global_order[i][j-1])
                    if(j==1):
                        whole_routes.append(temp)
                        break
                else:
                    if final_list[-1] != 'x':
                        final_list.append('x')
                    whole_routes.append(temp)
                    temp=[self.global_order[i][j-1]]
                    if(j==1):
                        break
            # Improve self.global_order  new
            for j in range(len(self.global_order)):
                for k in range(len(self.global_order[j])):
                    if dest==self.global_order[j][k][1] and self.global_order[j][k] not in final_list:
                        final_list.insert(0,self.global_order[j][k])

            #supplement
            for j in range(len(whole_routes)):
                if whole_routes[j][-1]!='checked':
                    if(whole_routes[j][0][1]!=dest):
                        tempx=-1
                        for x in range(len(final_list)):
                            if(whole_routes[j][0][1]==final_list[x][0]):
                                tempx=x
                                break
                        for x in range(tempx,-1,-1):
                            whole_routes[j].insert(0,final_list[x])
                    whole_routes[j].append('checked')
        #reverse the list, assign the self.execute_routes variables
        for i in range(len(whole_routes)):
            temp=[]
            whole_routes[i].pop()
            for j in range(len(whole_routes[i])-1,-1,-1):
                temp.append(whole_routes[i][j])
            self.execute_routes.append(temp)

        ###find this limit the performance, so delete
        #check the duplications
        # for i in range(len(self.execute_routes)):
        #     for j in range(i+1,len(self.execute_routes)):
        #         if(i!=j):
        #             if self.execute_routes[i] in self.execute_routes[j]:
        #                 if self.execute_routes[i] in self.execute_routes:# Avoid duplication
        #                     self.execute_routes.remove(self.execute_routes[i])
        #                 if self.execute_routes[j] not in self.execute_routes:
        #                     self.execute_routes.append(self.execute_routes[j])
        #             elif self.execute_routes[j] in self.execute_routes[i]:
        #                 if self.execute_routes[j] in self.execute_routes:# Avoid duplication
        #                     self.execute_routes.remove(self.execute_routes[j])
        #                 if self.execute_routes[i] not in self.execute_routes:
        #                     self.execute_routes.append(self.execute_routes[i])
        #             else:
        #                 if self.execute_routes[i] not in self.execute_routes:
        #                     self.execute_routes.append(self.execute_routes[i])
        #                 if self.execute_routes[j] not in self.execute_routes:
        #                     self.execute_routes.append(self.execute_routes[j])

        # avoid duplication
        for i in range(len(self.execute_routes) - 1, -1, -1):
            for j in self.executed:
                if self.execute_routes[i] == j:
                    del self.execute_routes[i]
                    break

        #check the startStack
        # check blocks'stk (recursion dao 0,stk>0 de qingkuang ZHONGWEN)
        for i in range(len(self.execute_routes)-1,-1,-1):
            current_stks = 0
            check_exe = True
            if(current_stks<self.nodes[self.execute_routes[i][0][0]].startStack):#check the first block
                check_exe=False
            current_stks = self.nodes[self.execute_routes[i][0][0]].addStack
            for j in range(len(self.execute_routes[i])):
                print("index{} start{} add{}".format(self.execute_routes[i][j][1],self.nodes[self.execute_routes[i][j][1]].startStack,self.nodes[self.execute_routes[i][j][1]].addStack))
                if (current_stks < self.nodes[self.execute_routes[i][j][1]].startStack):
                    check_exe = False
                    if self.retryx==True and self.execute_routes[i][-1][1]==149:
                        self.retry(self.execute_routes[i],j,p)
                        return
                current_stks = current_stks + self.nodes[self.execute_routes[i][j][1]].addStack
            if (current_stks<0):#check the last block/source
                check_exe = False
            if(check_exe==False): # ensure that blocks can be executed successfully
                del self.execute_routes[i]


    def build_CFG(self):
        #compute all block indexes, which relate to the location of instructions
        for index in range(len(self.nodes)):
            temp=0
            for i in range(len(self.nodes[index].bytecode)):
                temp=temp+len(self.nodes[index].bytecode[i])
            if (index < len(self.nodes)-1):
                self.block_index.append(self.block_index[index]+int(temp/2))

        # Deal with easyJump situation
        for index in range(len(self.nodes)-1):#the last block is empty
            #Deal with the special situation
            if(len(self.nodes[index].bytecode)==1):
                edge=[]
                edge.append(self.block_index[index])
                edge.append(self.block_index[index+1]) #Note that index starts with 0
                edge.append("EA")
                self.edges.append(edge)
                continue
            # Deal with easyJump situation, like RETURN or RETURN or SELFDESTRUCT or STOP"
            if(self.nodes[index].bytecode[-1]=='fd' or self.nodes[index].bytecode[-1]=='f3'
                or self.nodes[index].bytecode[-1]=='ff'or self.nodes[index].bytecode[-1]=='00'):
                print(str(index)+" is related to REVERT or RETURN or SELFDESTRUCT or STOP")
                self.nodes[index].easyJump=True
                edge=[]
                edge.append(self.block_index[index])
                edge.append(self.block_index[len(self.nodes)-1]) #Note that index starts with 0
                edge.append("EA")
                self.edges.append(edge)
            # Deal with easyJump situation, like push/JUMP
            if(self.nodes[index].bytecode[-2][0]=='6' or self.nodes[index].bytecode[-2][0]=='7' ):
                if(self.nodes[index].bytecode[-1]=='56' or self.nodes[index].bytecode[-1]=='57'):
                    print(str(index)+" is related to PUSH/JUMP(I) type ")
                    self.nodes[index].easyJump = True
                    temp=0
                    temp_count=1
                    for i in self.nodes[index].bytecode[-2][2:]:
                        pow_result=pow(16,len(self.nodes[index].bytecode[-2][2:])-temp_count)
                        temp=temp+pow_result*str2int(i)
                        temp_count=temp_count+1
                    edge = []
                    edge.append(self.block_index[index])
                    edge.append(temp)  # Because the index starts with 0
                    edge.append("EA")
                    self.edges.append(edge)
                    # JUMPI need to process with other edge: pc+1
                    if(self.nodes[index].bytecode[-1]=='57'):
                        edge2=[]
                        edge2.append(self.block_index[index])
                        edge2.append(self.block_index[index+1])
                        edge2.append("EA")
                        self.edges.append(edge2)
            #for complex jump:when we meet JUMPI, we need to add "pc+1"
            if (self.nodes[index].bytecode[-2][0] != '6' and self.nodes[index].bytecode[-2][0] != '7'):
                if (self.nodes[index].bytecode[-1] == '57'):
                    edge2=[]
                    edge2.append(self.block_index[index])
                    edge2.append(self.block_index[index+1])
                    edge2.append("EA")
                    self.edges.append(edge2)

        #For test
        test_e=count_easyJump(self.edges)

        # Deal with complexJump situation
        #complexpoint=0 #record the number of points with complex situations
        #resolvepoint=0 #record the number of points that are already computed
        point_list = []
        self.retryx=False
        for index in range(len(self.nodes) - 1):  # the last block is empty
            if self.nodes[index].easyJump==False:
                #complexpoint=complexpoint+1
                print(str(index)+" COMPLEX")
                if(self.nodes[index].startStack<=0):# It can be solved by executing only a basic block
                    SA,only_A=self.execute_JUMPtarget(self.nodes[index].bytecode,self.block_index[index])
                    if only_A!='un':#not unresolved jump
                        self.nodes[self.block_index.index(SA)].onlyAmode=only_A
                    #resolvepoint = resolvepoint + 1
                    self.nodes[index].resolved=True
                #new change 2023/01/23
                # else:
                #     str_stk=self.nodes[index].startStack
                #     #this function generates the execution order
                #     self.T=0
                #     self.check_stk(index,str_stk)
                #     if self.execute_order!=[]:
                #         self.global_order.append(self.execute_order)
                #     self.execute_order = []

        #while(complexpoint!=resolvepoint):
        first=True
        unsetpoint=[]
        for index in range(len(self.nodes) - 1):  # the last block is empty
            if self.nodes[index].easyJump == False and self.nodes[index].resolved == False:
                unsetpoint.append(index)
        for index in range(len(self.nodes) - 1):  # the last block is empty
            if index==149:
                print()
            if self.nodes[index].easyJump == False and self.nodes[index].resolved == False:
                #New change 2023/01/23
                #-------------------
                # if(first==False):#after first loop, execute this part
                str_stk = self.nodes[index].startStack
                # this function generates the execution order
                self.T=0
                self.check_stk(index, str_stk)
                if self.execute_order != []:
                    self.global_order.append(self.execute_order)
                self.execute_order = []

                #Generate the execute routes
                self.generate_routes(len(point_list))
                # -------------------

                #execute instructions by executing routes
                for number in range(len(self.execute_routes)):
                    execute_blocks=self.nodes[self.execute_routes[number][0][0]].bytecode
                    for block_n in range(len(self.execute_routes[number])):
                        execute_blocks=execute_blocks+self.nodes[self.execute_routes[number][block_n][1]].bytecode
                    SA,only_A=self.execute_JUMPtarget(execute_blocks,self.block_index[self.execute_routes[number][-1][1]])
                    if only_A != 'un':  # not unresolved jump
                        self.nodes[self.block_index.index(SA)].onlyAmode = only_A

                    point_list.append(self.execute_routes[number][-1][1])
                    self.executed.append(self.execute_routes[number])
                    #resolvepoint = resolvepoint + 1
                self.execute_routes=[]

                #control conditions
                point_list=list(set(point_list))
                for number in point_list:
                    self.nodes[number].resolved = True
                    if number in unsetpoint:
                        unsetpoint.remove(number)

                first=False

        #new change2023/01/23
        mcount=0
        self.retryx=True
        for index in range(len(unsetpoint)):
            if unsetpoint[index]==149:
                print()
            str_stk = self.nodes[unsetpoint[index]].startStack
            # this function generates the execution order
            self.T = 0
            self.check_stk(unsetpoint[index], str_stk)
            if self.execute_order != []:
                self.global_order.append(self.execute_order)
                mcount=mcount+1
            self.execute_order = []

            # Generate the execute routes
            self.generate_routes(29+mcount)

            # execute instructions by executing routes
            for number in range(len(self.execute_routes)):
                execute_blocks = self.nodes[self.execute_routes[number][0][0]].bytecode
                for block_n in range(len(self.execute_routes[number])):
                    execute_blocks = execute_blocks + self.nodes[self.execute_routes[number][block_n][1]].bytecode
                SA, only_A = self.execute_JUMPtarget(execute_blocks,
                                                     self.block_index[self.execute_routes[number][-1][1]])
                if only_A != 'un':  # not unresolved jump
                    self.nodes[self.block_index.index(SA)].onlyAmode = only_A

                point_list.append(self.execute_routes[number][-1][1])
                self.executed.append(self.execute_routes[number])
                # resolvepoint = resolvepoint + 1
            self.execute_routes = []

            # control conditions
            point_list = list(set(point_list))
            for number in point_list:
                self.nodes[number].resolved = True

        #this step is to connect the blocks
        for edge in self.edges:
            ee=[]
            ee.append(self.block_index.index(edge[0]))
            try:
                ee.append(self.block_index.index(edge[1]))
            except ValueError:
                if edge[1] > self.block_index[-1]:
                    ee.append(edge[1])
                    ee.append('Max')# it can mark this edge points to other place (over limit)
                    self.nodes[self.block_index.index(edge[0])].edges.append(ee)
                else:
                    for index in range(len(self.block_index)):
                        if (self.block_index[index]<edge[1] and self.block_index[index+1]>edge[1]):
                            ee.append(index)
                            ee.append('Mid')# it can mark this edge points to the middle part of this block
                            self.nodes[self.block_index.index(edge[0])].edges.append(ee)
                            #this part needs to record the reverse edges
                            execute_sub=edge[1] - self.block_index[index]
                            exe_point=0
                            for i in self.nodes[index].bytecode:
                                if execute_sub <= 0:
                                    break
                                if i[0]=='6' or i[0]=='7':
                                    temp=(str2int(i[0])-6)*16+str2int(i[1])
                                    execute_sub = execute_sub - 2-temp
                                    pass
                                else:
                                    execute_sub=execute_sub-1
                                exe_point=exe_point+1
                            self.nodes[index].Revedges.append([self.block_index.index(edge[0]),exe_point])
                            break
            else:
                self.nodes[self.block_index.index(edge[0])].edges.append(ee)
            print()
        print("look_up")


