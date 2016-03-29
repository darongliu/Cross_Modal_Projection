"""
from all image file to get all relation and entity information
"""
import os
try:
   import cPickle as pickle
except:
   import pickle

root_file = "../images"
dump_path = "../metadata/all_entity"

all_file = os.listdir(root_file)

idx2name = dict()
name2idx = dict()
entity = []
relation = []

entity_temp = set()
relation_temp = set()
for f in all_file :
    triplet = f.split(" " , 2)

    if len(triplet) == 3 :
        entity_temp.add(triplet[0])
        entity_temp.add(triplet[2])
        relation_temp.add(triplet[1])

entity = list(entity_temp)
relation  = list(relation_temp)

count = 0
for e in entity :
    idx2name[count] = e
    name2idx[e] = count
    count = count + 1
for r in relation :
    idx2name[count] = r
    name2idx[r] = count
    count = count + 1

all_result = [entity,relation,idx2name,name2idx]
pickle.dump(all_result , open(dump_path , 'wb'))


