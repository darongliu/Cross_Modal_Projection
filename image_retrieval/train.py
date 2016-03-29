import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat
import numpy as np
import argparse

from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

import triplet_encoding
import image_project
import file_handle as F

try:
   import cPickle as pickle
except:
   import pickle
import sys

np.random.seed(1234)

def make_train(image_size , word_size , first_hidden_size , proj_size , reg_lambda) :
    #initialize model
    P = Parameters()
    image_projecting = image_project.build(P, image_size, proj_size)
    batched_triplet_encoding , vector_triplet_encoding = triplet_encoding.build(P , word_size , first_hidden_size , proj_size)   

    image_vector = T.vector()

    #training
    correct_triplet =  [T.vector(dtype='float32') , T.vector(dtype='float32') , T.vector(dtype='float32')] #[E,R,E]
    negative_triplet = [T.matrix(dtype='float32') , T.matrix(dtype='float32') , T.matrix(dtype='float32')]

    image_projection_vector = image_projecting(image_vector)
    image_projection_matrix = repeat(image_projection_vector.dimshuffle(('x',0)) , negative_triplet[0].shape[0] , axis=0)
    correct_triplet_encoding_vector = vector_triplet_encoding(correct_triplet[0] , correct_triplet[1] , correct_triplet[2])
    negative_triplet_encoding_matrix = batched_triplet_encoding(negative_triplet[0] , negative_triplet[1] , negative_triplet[2])

    correct_cross_dot_scalar = T.dot(image_projection_vector , correct_triplet_encoding_vector)
    negative_cross_dot_vector = T.batched_dot(image_projection_matrix , negative_triplet_encoding_matrix)

    #margin cost
    zero_cost = T.zeros_like(negative_cross_dot_vector)
    margin_cost = 1 - correct_cross_dot_scalar + negative_cross_dot_vector
    cost_vector = T.switch(T.gt(zero_cost , margin_cost) , zero_cost , margin_cost)

    #regulizar cost
    params = P.values()
    l2 = T.sum(0)
    for p in params:
        l2 = l2 + (p ** 2).sum()        
    cost = T.sum(cost_vector)/T.shape(negative_triplet[0])[0] + reg_lambda * l2 #assume word vector has been put into P #unsolved
    grads = [T.clip(g, -100, 100) for g in T.grad(cost, wrt=params)]

    lr = T.scalar(name='learning rate',dtype='float32')
    train = theano.function(
        inputs=[image_vector, correct_triplet[0], correct_triplet[1], correct_triplet[2], negative_triplet[0], negative_triplet[1], negative_triplet[2], lr],
        outputs=cost,
        updates=updates.rmsprop(params, grads, learning_rate=lr),
        allow_input_downcast=True
    )

    #valid
    valid = theano.function(
        inputs=[image_vector, correct_triplet[0], correct_triplet[1], correct_triplet[2], negative_triplet[0], negative_triplet[1], negative_triplet[2]],
        outputs=cost,
        allow_input_downcast=True

    )
    #visualize
    image_project_fun = theano.function(
        inputs=[image_vector],
        outputs=image_projection_vector,
        allow_input_downcast=True
    )
    #testing
    all_triplet = [T.matrix(dtype='float32') , T.matrix(dtype='float32') , T.matrix(dtype='float32')]
    image_projection_matrix_test = repeat(image_projection_vector.dimshuffle(('x',0)) , all_triplet[0].shape[0] , axis=0)
    all_triplet_encoding_matrix = batched_triplet_encoding(all_triplet[0] , all_triplet[1] , all_triplet[2])
    all_cross_dot_vector = T.batched_dot(image_projection_matrix_test , all_triplet_encoding_matrix)

    test = theano.function(
        inputs=[image_vector, all_triplet[0], all_triplet[1], all_triplet[2]],
        outputs=all_cross_dot_vector,
        allow_input_downcast=True

    )

    return P , train , valid , image_project_fun , test

def print_info(args) :
    print 'retrieval information'
    print '-------------------------------------------------------'
    print 'train data : %s'  % args.train[0]
    print 'train label : %s' % args.train[1]
    print 'test data: %s'    % args.test[0]
    print 'test label: %s'   % args.test[1] 
    print 'load file: %s' % args.load_net
    print '-------------------------------------------------------'    
def image_retrieve_straight(args) :
    print "start retrieve by original vector"
    train_data  = pickle.load(open(args.train[0],'rb'))
    train_label = pickle.load(open(args.train[1],'rb'))
    test_data   = pickle.load(open(args.test[0],'rb'))
    test_label  = pickle.load(open(args.test[1],'rb'))

    test_length = len(test_data)

    all_score = 0.
    for idx in range(test_length) :
        test_image_vector = test_data[idx]

        all_cosine = []
        for train_image_vector in train_data :
            if train_image_vector is None :
                all_cosine.append(-1*np.inf)
                continue
            dot_value = np.dot(train_image_vector,test_image_vector) 
            train_2_norm = np.linalg.norm(train_image_vector)
            test_2_norm  = np.linalg.norm(test_image_vector )
            cos_sim = dot_value/train_2_norm/test_2_norm
            result = cos_sim #normalize or not can discuss
            all_cosine.append(result)
        sort_idx = F.len_argsort(all_cosine)

        test_triplet = test_label[idx][0]
        image_score = 0.
        sim_function = F.label_sim
        for i in range(args.retrieve_num) :
            idx_optimal = sort_idx[-1*i-1]
            train_triplet = train_label[idx_optimal][0]
            label_sim = sim_function(train_triplet,test_triplet)
            image_score = image_score + label_sim
            print "test idx:",idx,"retrieve idx:",i,"label similarity:",label_sim

        avg_score = image_score/args.retrieve_num
        all_score = all_score + avg_score 
        print "test idx:",idx,"average score:",avg_score 

    print 'result score : %f' % (all_score/test_length)
    

def image_retrieve_projection(args) :
    print "start retrieve by projection\n\n"
    train_data  = pickle.load(open(args.train[0],'rb'))
    train_label = pickle.load(open(args.train[1],'rb'))
    test_data   = pickle.load(open(args.test[0],'rb'))
    test_label  = pickle.load(open(args.test[1],'rb'))

    test_length = len(test_data)

    print 'start initializing model'
    P, train, valid, image_project, test = make_train(
        image_size=1,
        word_size=1,
        first_hidden_size=1,
        proj_size=1,
        reg_lambda=1
    )
    P.load(args.load_net)

    all_score = 0.
    for idx in range(test_length) :
        test_image_vector = image_project(test_data[idx])

        all_cosine = []
        for train_image_vector_temp in train_data :
            if train_image_vector_temp is None :
                all_cosine.append(-1*np.inf)
                continue            
            train_image_vector = image_project(train_image_vector_temp)
            dot_value = np.dot(train_image_vector,test_image_vector) 
            train_2_norm = np.linalg.norm(train_image_vector)
            test_2_norm  = np.linalg.norm(test_image_vector )
            cos_sim = dot_value/train_2_norm/test_2_norm
            result = cos_sim #normalize or not can discuss
            all_cosine.append(result)
        sort_idx = F.len_argsort(all_cosine)

        test_triplet = test_label[idx][0]
        image_score = 0.
        sim_function = F.label_sim
        for i in range(args.retrieve_num) :
            idx_optimal = sort_idx[-1*i-1]
            train_triplet = train_label[idx_optimal][0]
            label_sim = sim_function(train_triplet,test_triplet)
            image_score = image_score + label_sim
            print "test idx:",idx,"retrieve idx:",i,"label similarity:",label_sim

        avg_score = image_score/args.retrieve_num
        all_score = all_score + avg_score 
        print "test idx:",idx,"average score:",avg_score 

    print 'result score : %f' % (all_score/test_length)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DESCRIPTION')
    args = F.parse_args(parser)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    print_info(args)
    image_retrieve_straight(args)
    image_retrieve_projection(args)
