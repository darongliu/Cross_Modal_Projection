import argparse
import numpy as np
import pickle

def parse_args(parser):
    parser.add_argument('--train', nargs=2, action="store", metavar="FILE", help='training data/label !')
    parser.add_argument('--test' , nargs=2, action="store", metavar="FILE", help='testing data/label!')
    parser.add_argument('--word-vector', action="store", dest='word_vector', help='word vector path ' ,default=None)
    parser.add_argument('--all-info', action="store", dest='all_info', help='all entity and relation info' ,default=None)
    parser.add_argument('--learning-rate', action="store", dest='lr', help='learing rate at begining, default: 0.01 ', type=float, default=0.01)
    parser.add_argument('--hidden-size', action="store", dest='first_hidden_size', help='first hidden size, default: 50 ', type=int, default=50)
    parser.add_argument('--proj-size', action="store", dest='proj_size', help='projection size, default: 10 ', type=int, default=10)
    parser.add_argument('--reg-lambda', action="store", dest='reg_lambda', help='regulization propotion', type=float, default=0.01)    
    parser.add_argument('--improvement-rate', action="store", dest='improvement_rate', help='relative improvement for early stopping on ppl , default: 0.005 ', type=float, default=0.005)
#    parser.add_argument('--minibatch-size', action="store", dest='minibatch_size', help='minibatch size for training, default: 100', type=int, default=100)
    parser.add_argument('--max-epoch', action="store", dest='max_epoch', help='maximum number of epoch if not early stopping, default: 1000', type=int, default=1000)
    parser.add_argument('--save-net', action="store", dest="save_net", default=None, metavar="FILE",help="Save RNN to file")
    parser.add_argument('--load-net', action="store", dest="load_net", default=None, metavar="FILE", help="Load RNN from file")

    return parser.parse_args()  
  
def get_image_dim(train_data) :
    return train_data[0].shape[0] #assume data is stored in numpy array

def get_word_dim(word_vector) :
    return word_vector[word_vector.keys()[0]].shape[0]

def label2vector(triplet , all_info , word_vector) :
    def number2vector(num , all_info , word_vector) :
        idx2name = all_info[2]
        name = idx2name[num]
        return word_vector[name]
    return [number2vector(triplet[0],all_info,word_vector),number2vector(triplet[1],all_info,word_vector),number2vector(triplet[2],all_info,word_vector)]

def combine(ori , add_vector) :
    result = []
    if len(ori) == 0 :
        for element in add_vector :
            result.append(element.reshape(1,element.shape[0]))
    else :
        for element_ori,element_add in zip(ori,add_vector) :
            new = element_add.reshape(1,element_add.shape[0])
            com = np.concatenate((element_ori,new),axis=0)
            result.append(com)
    return result

def negative_generator(triplet , all_info , word_vector) :
    all_entity   = all_info[0]
    all_relation = all_info[1]
    idx2name     = all_info[2]
    name2idx     = all_info[3]

    result = []
    for entity in all_entity :
        if name2idx[entity] != triplet[0] :
            neg_triplet = [name2idx[entity],triplet[1],triplet[2]]
            neg_vector  = label2vector(neg_triplet,all_info,word_vector)
            result = combine(result,neg_vector)  

    for relation in all_relation :
        if name2idx[relation] != triplet[1] :
            neg_triplet = [triplet[0],name2idx[relation],triplet[2]]
            neg_vector  = label2vector(neg_triplet,all_info,word_vector)
            result = combine(result,neg_vector)

    for entity in all_entity :
        if name2idx[entity] != triplet[2] :
            neg_triplet = [triplet[0],triplet[1],name2idx[entity]]
            neg_vector  = label2vector(neg_triplet,all_info,word_vector)
            result = combine(result,neg_vector)

    return result

def one_change_test(triplet,all_info,word_vector,pos) :
    all_entity   = all_info[0]
    all_relation = all_info[1]
    idx2name     = all_info[2]
    name2idx     = all_info[3]

    result = []
    count  = 0

    if pos == 0 :
        for entity in all_entity :
            test_triplet = [name2idx[entity],triplet[1],triplet[2]]
            test_vector  = label2vector(test_triplet,all_info,word_vector)
            result = combine(result,test_vector)  

            if name2idx[entity] == triplet[0] :
                answer = count
            count = count + 1

    if pos == 1 :
        for relation in all_relation :
            test_triplet = [triplet[0],name2idx[relation],triplet[2]]
            test_vector  = label2vector(test_triplet,all_info,word_vector)
            result = combine(result,test_vector)

            if name2idx[relation] == triplet[1] :
                answer = count
            count = count + 1

    if pos == 2 :
        for entity in all_entity :
            test_triplet = [triplet[0],triplet[1],name2idx[entity]]
            test_vector  = label2vector(test_triplet,all_info,word_vector)
            result = combine(result,test_vector)

            if name2idx[entity] == triplet[2] :
                answer = count
            count = count + 1


    return result , answer

def binary_score(test_score,answer) :
    test_answer = np.argmax(test_score)
    if test_answer == answer :
        return 1
    else :
        return 0

def rank_score(test_score,answer) :
    rank = 1
    for idx in range(len(test_score)) :
        if idx == answer :
            continue
        if test_score[idx] > test_score[answer] :
            rank = rank + 1
    return 1./float(rank)


