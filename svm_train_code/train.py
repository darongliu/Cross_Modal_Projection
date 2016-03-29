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

    print 'start training'


    for i in range(3) :
        print 'load data pos' , i
        x = []
        y = []
        for image_vector , label in zip(train_data,train_label) :
            if image_vector is None :
                continue
            x.append([float(j) for j in list(image_vector)])
            y.append(int(label[0][i]))
        print type(x[0][0])
        print 'training svm pos' , i
        prob = svm_problem(y, x)
        param = svm_parameter('-t 0 -b 1')
        m = svm_train(prob, param)
        svm_save_model(args.save_net + str(i), m)


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

#    print 'left_p_label',left_p_label
#    print 'left_p_acc',left_p_acc
#    print 'left_p_val',left_p_val      
    score = 0
    guess_result = zip(left_p_label,relation_p_label,right_p_label) 
    for idx , result in enumerate(guess_result) :
        if int(result[0]) == test_label[idx][0][0] and int(result[1]) == test_label[idx][0][1] and int(result[2]) == test_label[idx][0][2] :
            score = score + 1
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
