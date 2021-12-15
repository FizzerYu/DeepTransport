import logging
import os
import pandas as pd
import numpy as np
import time
import collections
from tqdm import tqdm
import copy
from torch.utils.data.dataset import Dataset


class preProcessData():
    def __init__(self, args, trainTestSplitDate='2016-05-01') -> None:

        ##################################################
        # 1. prepare graph data
        self.radius_upStreamGraph, self.radius_downStreamGraph = self.loadFinalGraph(args.data_path, args.radius)
        logging.info("1. Successfully load graph data")
        ##################################################
        # 2. prepare traffic condition data
        traffic_condition = self.loadTrafficData(args.data_path)

        self.train = traffic_condition[traffic_condition['timestamp']<trainTestSplitDate].reset_index(drop=True)
        self.test = traffic_condition[traffic_condition['timestamp']>=trainTestSplitDate].reset_index(drop=True)
        logging.info("2. Successfully load traffic condition data")
        
    def loadFinalGraph(self, data_path, radius):
        upStreamGraph, downStreamGraph = self.loadGraph(data_path)

        radius_downStreamGraph = self.getRadiusStreamGraph(downStreamGraph, radius)
        radius_upStreamGraph = self.getRadiusStreamGraph(upStreamGraph, radius)
        radius_downStreamGraph = self.getRadiusStreamGraph(downStreamGraph, radius)
        radius_upStreamGraph = self.getRadiusStreamGraph(upStreamGraph, radius)

        ## fillna
        for k, item in radius_downStreamGraph.items():
            if item.shape[1]==0:
                radius_downStreamGraph[k] = radius_upStreamGraph[k]
        for k, item in radius_upStreamGraph.items():
            if item.shape[1]==0:
                radius_upStreamGraph[k] = radius_downStreamGraph[k]
        return radius_upStreamGraph, radius_downStreamGraph

    def loadGraph(self, gpath):
        """
        read graph file, build up and down stream graph
        """
        graphdata = pd.read_csv(os.path.join(gpath, "graph.csv") )
        downStreamGraph = collections.defaultdict(list)
        for i in range(len(graphdata)):
            downStreamGraph[ graphdata.iloc[i, 0] ].append( graphdata.iloc[i, 1] )
            
        upStreamGraph = collections.defaultdict(list)
        for i in range(len(graphdata)):
            upStreamGraph[ graphdata.iloc[i, 1] ].append( graphdata.iloc[i, 0] )
            
        logging.info("{}{}".format(len(downStreamGraph), len(upStreamGraph)))
        return upStreamGraph, downStreamGraph


    def getRadiusStreamGraph(self, StreamGraph, radius_Num):
        AUXradius_StreamGraph = {}
        for nowp in list(StreamGraph.keys()):
            AUXradius_StreamGraph[nowp] = []
            prevstation = set([nowp])
            # 第一次
            AUXradius_StreamGraph[nowp].append( [[x] for x in StreamGraph[nowp]] )
            prevstation.update( set(StreamGraph[nowp]) )
            for layer in range( radius_Num - 1 ):
                nextNodes = []
                prevNode = AUXradius_StreamGraph[nowp][-1]
                flattenPrevNode = [y for x in prevNode for y in x]
                for idx, prevp in enumerate(flattenPrevNode):   # 上一层的所有节点
                    tmp_pools = [x for x in StreamGraph[prevp] if x not in prevstation]  # 不去重会有loop
                    if tmp_pools == []: tmp_pools=[prevp]
        #             tmp_pools = [x for x in StreamGraph[prevp]] 
                    prevstation.update( set(tmp_pools) )
                    nextNodes.append( tmp_pools )

                AUXradius_StreamGraph[nowp].append( nextNodes )

        radius_StreamGraph = {}
        for numsNam in AUXradius_StreamGraph.keys():
            nums = AUXradius_StreamGraph[numsNam]
            new_nums = copy.deepcopy( nums )
            for layer in list(range( radius_Num - 1 ))[::-1]:
                flatL = [x for y in nums[layer] for x in y]
                newxtLen = [len(x) for x in new_nums[layer+1]]
                adjFlatl = [[] for y in nums[layer]]
                flag = 0
                for idx, v in enumerate( nums[layer] ):
                    for jj in v:
                        adjFlatl[idx].extend( [jj for k in range( newxtLen[flag] )] )
                        flag += 1
                new_nums[layer] = adjFlatl
            radius_StreamGraph[ numsNam ] = new_nums

        radius_StreamGraph = {k:np.array([[y for xx in k for y in xx] for k in nums]) for k,nums in radius_StreamGraph.items()}
        return radius_StreamGraph

    def loadTrafficData(self, path):
        traffic_condition = pd.read_csv(path + "traffic_condition.csv")
        traffic_condition = traffic_condition.sort_values(by='timestamp').reset_index(drop=True)

        traffic_condition['timestamp'] = traffic_condition['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)) )
        traffic_condition['timestamp'] = pd.to_datetime(traffic_condition['timestamp'])
                                                                                
        traffic_condition = traffic_condition.sort_values(by='timestamp').reset_index(drop=True)

        data = []
        tmpdata = pd.DataFrame(traffic_condition['timestamp'].drop_duplicates().reset_index(drop=True))
        for i in traffic_condition['id'].drop_duplicates().tolist():
            tmpdata['id'] = i
            data.append( tmpdata.copy(deep=True) )
        data = pd.concat(data, axis=0)[['id', 'timestamp']]
        traffic_condition = traffic_condition.merge(data, on = ['id','timestamp'], how='right', sort=True)
        traffic_condition = traffic_condition.sort_values(by=['id','timestamp']).reset_index(drop=True)

        traffic_condition['condition'] = traffic_condition['condition'].fillna(method='pad')
        traffic_condition['limit_level'] = traffic_condition['limit_level'].fillna(method='pad')

        traffic_condition['NT'] = (traffic_condition['timestamp'].astype(int)/ 100000000000).astype(int)
        traffic_condition['timeId'] = traffic_condition['id'].astype(str) + "+" + traffic_condition['NT'].astype(str)
        return traffic_condition



def prepareData(df, pnum, ):
    ans = {}         # key: uid+time
    uids = df.id.drop_duplicates().tolist()
    for uid in tqdm(uids):
        tmpudf = df.query(f"id=={uid}").reset_index(drop=True)
        tivF = [tmpudf.loc[0, 'limit_level']]
        for idx in range(len(tmpudf)-1):
            if idx == 0:
                tvO = [tmpudf.loc[0, 'condition']] * pnum
            elif pnum>idx:     # 最开始几个 需要补齐
                prev = tmpudf.loc[:pnum, 'condition'].tolist()
                repeat = pnum // (idx+1) + 1
                tvO = ( tmpudf.loc[:idx, 'condition'].tolist() * repeat )[:pnum]
            elif pnum<=idx:
                tvO = tmpudf.loc[(idx-pnum+1):idx, 'condition'].tolist()
            label = tmpudf.loc[idx+1, 'condition']
            timeuid = tmpudf.loc[idx, 'timeId']
            ans[ timeuid ] = [tvO + tivF, label]  # 这里合并了 self_tvO, self_tivF
    return ans



class DataSet(Dataset):
    def __init__(self, data, upgraph, downgraph, rnum, maxlen = 20):
        self.data = data
        self.uidTime = list( self.data.keys() )
        self.upgraph = upgraph
        self.downgraph = downgraph
        self.maxlen = maxlen
        self.rnum = rnum
        
    def __getitem__(self, index):
        
        # target location
        timeuid = self.uidTime[index]
        uid, time = timeuid.split("+")
        uid, time = int(uid), int(time)
        
        self_tvFeats, label = self.data[ timeuid ]
        
        ## upstream module
        ### select
        upSlot = self.upgraph[uid]
        if upSlot.shape[1] >= self.maxlen:
            upUids = upSlot[:, :self.maxlen]
        else:
            repeat = self.maxlen // upSlot.shape[1] + 1
            upUids = np.repeat(upSlot, repeat, axis = 1)[:, :self.maxlen]
        ### broadcast
        upSlots = []
        # for layer in range( self.rnum ):
        for layer in range( self.rnum ):
            upSlots.append( [self.data[f"{uid}+{time}"][0] for uid in upUids[layer]] )

            
        ## downstream module
        downSlot = self.downgraph[uid]
        if downSlot.shape[1] >= self.maxlen:
            downUids = downSlot[:, :self.maxlen]
        else:
            repeat = self.maxlen // downSlot.shape[1] + 1
            downUids = np.repeat(downSlot, repeat, axis = 1)[:, :self.maxlen]
        ### broadcast
        downSlots = []
        # for layer in range( self.rnum ):
        for layer in range( self.rnum ):
            downSlots.append( [self.data[f"{uid}+{time}"][0] for uid in downUids[layer]] )
        
        return np.array(self_tvFeats), np.array(upSlots), np.array(downSlots), label
        
    def __len__(self):
        return len( self.uidTime )
    