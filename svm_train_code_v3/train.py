import numpy as np
import argparse

import file_handle as F

import sys
sys.path.append('../libsvm/python')
from svmutil import *

try:
   import cPickle as pickle
except:
   import pickle
import sys

np.random.seed(1234)

def training(args) :
    train_data  = pickle.load(open(args.train[0],'rb'))
    train_label = pickle.load(open(args.train[1],'rb'))

    # training information
    print 'training information'
    print '-------------------------------------------------------'
    print 'examples in training file: %d' % len(train_data)
    print 'train data : %s' % args.train[0]
    print 'save file: %s' % args.save_net
    print '-------------------------------------------------------'

    print 'construct possible triplet tree'
    all_triplet = dict()
    for label_temp in train_label :
        label = label_temp[0]
        if label[0] not in all_triplet :
            all_triplet[label[0]] = dict()
        temp = all_triplet[label[0]]
        if label[1] not in temp :
            temp[label[1]] = dict()
        temp = temp[label[1]]
        if label[2] not in temp :
            temp[label[2]] = dict()
        temp = temp[label[2]]
    pickle.dump(all_triplet,open(args.save_net+'_tri_tree','wb'))

    print 'start training'

    label_info = []
    for i in range(3) :
        print 'load data pos' , i
        x = []
        y = []

        label2idx = dict()
        idx_count = 0

        for image_vector , label in zip(train_data,train_label) :
            if image_vector is None :
                continue
            x.append([float(j) for j in list(image_vector)])
            y.append(int(label[0][i]))

            if int(label[0][i]) not in label2idx :
                label2idx[int(label[0][i])] = idx_count
                idx_count = idx_count + 1
        label_info.append(label2idx)

        print type(x[0][0])
        print 'training svm pos' , i
        prob = svm_problem(y, x)
        param = svm_parameter('-t 0 -b 1')
        m = svm_train(prob, param)
        svm_save_model(args.save_net + str(i), m)

    pickle.dump(label_info,open(args.save_net+'_label_info' , 'wb'))


def testing(args) :
    test_data  = pickle.load(open(args.test[0],'rb'))
    test_label = pickle.load(open(args.test[1],'rb'))

    #testing information
    print 'testing information'
    print '-------------------------------------------------------'
    print 'test data: %s'  % args.test[0]
    print 'test label: %s' % args.test[1]
    print 'load file: %s' % args.load_net
    print '-------------------------------------------------------'

    m_left     = svm_load_model(args.load_net + '0')
    m_relation = svm_load_model(args.load_net + '1')
    m_right    = svm_load_model(args.load_net + '2')
    label_info = pickle.load(open(args.load_net+'_label_info','rb'))
    tri_tree   = pickle.load(open(args.load_net+'_tri_tree','rb'))

    test_length = len(test_data)

    test_generator = F.one_change_test
    test_score     = F.binary_score
#    test_score     = F.rank_score
    all_score = 0

    all_image = []
    left_label = []
    relation_label = []
    right_label = []
    for idx in range(test_length) :
        image_vector = test_data[idx]
        if image_vector is None :
            continue
        all_image.append([float(j) for j in list(image_vector)])
        left_label.append(int(test_label[idx][0][0]))
        relation_label.append(int(test_label[idx][0][1]))
        right_label.append(int(test_label[idx][0][2]))

    left_p_label, left_p_acc, left_p_val = svm_predict(left_label, all_image, m_left, '-b 1') 
    relation_p_label, relation_p_acc, relation_p_val = svm_predict(relation_label, all_image, m_relation, '-b 1') 
    right_p_label, right_p_acc, right_p_val = svm_predict(right_label, all_image, m_right, '-b 1') 


    score = 0
    """       
    guess_result = zip(left_p_label,relation_p_label,right_p_label) 
    for idx , result in enumerate(guess_result) :
        if int(result[0]) == test_label[idx][0][0] and int(result[1]) == test_label[idx][0][1] and int(result[2]) == test_label[idx][0][2] :
            score = score + 1
    """
    for idx in range(len(test_label)) :
        answer = test_label[idx][0]
        rank_left , bigger_label_left = F.one_rank(label_info[0],left_p_val[idx],answer[0])
        rank_relation , bigger_label_relation = F.one_rank(label_info[1],relation_p_val[idx],answer[1])
        rank_right , bigger_label_right = F.one_rank(label_info[2],right_p_val[idx],answer[2])
        invalid_num = F.unexist_tri_num(tri_tree , bigger_label_left , bigger_label_relation , bigger_label_right)
        total_rank = rank_left*rank_relation*rank_right - invalid_num
        one_score = 1./total_rank
        print one_score
        score = score + one_score

    print 'result score : %f' % score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DESCRIPTION')
    args = F.parse_args(parser)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    if args.train:
        training(args)
    elif args.test:
        testing(args)
