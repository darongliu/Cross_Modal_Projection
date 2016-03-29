import os
from random 
try:
   import cPickle as pickle
except:
   import pickle

root_file = "../images"
all_entity_path = "../metadata/all_entity"
dump_path = "../metadata/"

train_label = []
test_label  = []

train_path = []
test_path  = []

all_entity = pickle.load(open(all_entity_path,'rb'))
name2idx = all_entity[3]
del all_entity

all_file = os.listdir(root_file)

for f in all_file :
    if f[0] == ".":
        continue

    image_dir = os.path.join(root_file , f)
    images = os.listdir(image_dir)

    test_num = len(images)/3
    test_idx = random.sample(range(len(images)),test_num)

    for idx , name in enumerate(images) :
        if idx in test_idx :
            test_path.append(os.path.join(f,name))
            triplet = f.split(" " , 2)
            label = [name2idx[triplet[0]],name2idx[triplet[1]],name2idx[triplet[2]]]
            image_labels = [label]
            test_label.append(image_labels)
            print "test " , os.path.join(f,name)
        else :
            train_path.append(os.path.join(f,name))
            triplet = f.split(" " , 2)
            label = [name2idx[triplet[0]],name2idx[triplet[1]],name2idx[triplet[2]]]
            image_labels = [label]
            train_label.append(image_labels)
            print "train" , os.path.join(f,name)

pickle.dump(train_label , open(dump_path+'train_label' , 'wb'))
pickle.dump(test_label , open(dump_path+'test_label' , 'wb'))
pickle.dump(train_path , open(dump_path+'train_path' , 'wb'))
pickle.dump(test_path , open(dump_path+'test_path' , 'wb'))
