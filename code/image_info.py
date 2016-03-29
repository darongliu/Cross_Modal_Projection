"""
based on parsed training and testing info to generate some statiscal result
"""
import os
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

def connection_num(root_path , name , pos) :
    """
    root_path:image root directory
    name:counting entity(relation) name
    pos:position id
    """
    connect_element_pre = set()
    connect_element_nex = set()

    all_file = os.listdir(root_path)
    for f in all_file :
        triplet = f.split(" " , 2)
        if len(triplet) != 3 :
            continue
        if triplet[pos] == name :
            if pos == 1 :
                connect_element_pre.add(triplet[0])
                connect_element_nex.add(triplet[2])
            else :
                connect_element_pre.add(triplet[1])

    if pos == 1 :
        return float(len(connect_element_pre)),float(len(connect_element_nex))
    else :
        return float(len(connect_element_pre))

def avg_connection_num(root_path , all_entity_path) :
    all_info = pickle.load(open(all_entity_path,'rb'))
    all_entity = all_info[0]
    all_relation = all_info[1]

    count = 0
    score = 0.
    pos1 = []
    pos2 = []
    for entity in all_entity :
        connection1 = connection_num(root_path,entity,0)
        connection2 = connection_num(root_path,entity,2)
        pos1.append(connection1)
        pos2.append(connection2)

        score = score + (connection1 + connection2)/2.
        count = count + 1
        
        print entity , connection1 , connection2
    entity1 = (np.average(pos1),np.std(pos1))
    entity2 = (np.average(pos2),np.std(pos2))

    pos1 = []
    pos2 = []
    for relation in all_relation :
        connection1,connection2 = connection_num(root_path,relation,1)
        pos1.append(connection1)
        pos2.append(connection2)

        score = score + (connection1+connection2)/2
        count = count + 1

        print relation , connection1 , connection2
 
    relation1 = (np.average(pos1),np.std(pos1))
    relation2 = (np.average(pos2),np.std(pos2))

    return score / count , entity1 , entity2 , relation1 , relation2

if __name__ == "__main__":
    result , entity1 , entity2 , relation1 , relation2 = avg_connection_num('../images' , '../metadata/all_entity')

    print 'avg' , result , entity1 , entity2 , relation1 , relation2
