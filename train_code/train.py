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

    return P , train , valid , test

def training(args) :
    all_info    = pickle.load(open(args.all_info,'rb'))

    train_data  = pickle.load(open(args.train[0],'rb'))
    train_label = pickle.load(open(args.train[1],'rb'))

    word_vector = pickle.load(open(args.word_vector,'rb'))

    # training information
    print 'training information'
    print '-------------------------------------------------------'
    print 'examples in training file: %d' % len(train_data)
    print 'train data : %s' % args.train[0]
#    print 'train label: %s' % args.valid[1]
    print 'learning rate: %f' % args.lr
#    print 'minibatch size: %d' % args.minibatch_size
    print 'max epoch: %d' % args.max_epoch
    print 'improvement rate: %f' % args.improvement_rate
    print 'save file: %s' % args.save_net
    print '-------------------------------------------------------'

    print 'start initializing model'

    image_dim = F.get_image_dim(train_data)
    word_dim  = F.get_word_dim(word_vector)

    P , train, valid , test = make_train(
        image_size=image_dim,
        word_size =word_dim,
        first_hidden_size=args.first_hidden_size,
        proj_size=args.proj_size,
        reg_lambda=args.reg_lambda
    )

    print 'start training'
    learning_rate = args.lr
#    min_valid_cost = float('inf')

    train_length = len(train_data)
#    valid_length = len(valid_data)
    min_train_cost = np.inf
    for epoch in range(args.max_epoch):
        print "\nepoch %d" % epoch
        train_cost = 0

        for idx in range(train_length) :
            if train_data[idx] == None :
                continue
            image_vector = train_data[idx]
            triplet = train_label[idx][0]
            correct_triplet  = F.label2vector(triplet,all_info,word_vector)
            negative_triplet = F.negative_generator(triplet,all_info,word_vector)

            cost = train(image_vector,correct_triplet[0],correct_triplet[1],correct_triplet[2],negative_triplet[0],negative_triplet[1],negative_triplet[2],learning_rate) 
            train_cost = train_cost + cost
            sys.stdout.write( '\r%d train idx / %d total train samples, cost: %f '% (idx+1, train_length, cost) )
            sys.stdout.flush()

        train_cost = train_cost/train_length
        print "\ntrain cost: %f" % train_cost

        if train_cost < min_train_cost :
            min_train_cost = train_cost
            P.save(args.save_net)
            print "save best model"
            continue

        #here we do not adopt early stop

        elif (train_cost - min_train_cost) / min_train_cost > args.improvement_rate:
            print 'change learning rate from %f to %f' % (learning_rate, learning_rate/2)
            learning_rate = learning_rate / 2.
"""
        #valid
        valid_cost = 0
        for idx in range(valid_length) :
            image_vector = valid_data[idx]
            triplet = valid_label[idx]
            correct_triplet = label2vector(triplet)
            all_negative_triplet = all_negative_generator(triplet,entity,relation,...)

            cost = valid(image_vector,correct_triplet,all_negative_triplet) 
            valid_cost = valid_cost + cost

        valid_cost = valid_cost/valid_length
        print "\ntrain cost: %f, valid cost: %f" % (train_cost, valid_cost)


        if valid_cost < min_valid_cost:
            min_valid_cost = valid_cost
            P.save(args.save_net)
            print "save best model"
            continue

        #here we do not adopt early stop

        elif (valid_cost - min_valid_cost) / min_valid_cost > args.improvement_rate:
            print 'change learning rate from %f to %f' % (learning_rate, learning_rate/2)
            learning_rate = learning_rate / 2.
"""

def testing(args) :
    test_data  = pickle.load(open(args.test[0],'rb'))
    test_label = pickle.load(open(args.test[1],'rb'))

    word_vector = pickle.load(open(args.word_vector,'rb'))
    all_info    = pickle.load(open(args.all_info,'rb'))

    print 'start initializing model'
    P , train, valid , test = make_train(
        image_size=1,
        word_size=1,
        first_hidden_size=1,
        proj_size=1,
        reg_lambda=1
    )
    P.load(args.load_net)

    #testing information
    print 'testing information'
    print '-------------------------------------------------------'
    print 'test data: %s'  % args.test[0]
    print 'test label: %s' % args.test[1]
    print 'load file: %s' % args.load_net
    print '-------------------------------------------------------'

    test_length = len(test_data)

    test_generator = F.all_test
#    test_generator = F.one_change_test
#    test_score     = F.binary_score
    test_score     = F.rank_score
    all_score = 0
    for idx in range(test_length) :
        image_vector = test_data[idx]
        triplet = test_label[idx][0]
        """        
        #for one_change_test
        for pos in range(3) :
            test_example , answer= test_generator(triplet,all_info,word_vector,pos)

            result = test(image_vector , test_example[0],test_example[1],test_example[2])
            score = test_score(result , answer)
            all_score = all_score + score

            print "test idx:",idx,"position:",pos,"score:",score
        """
        #for all_test
        test_example , answer= test_generator(triplet,all_info,word_vector)
        result = test(image_vector , test_example[0],test_example[1],test_example[2])
        score = test_score(result , answer)
        all_score = all_score + score
        print "test idx:",idx,"score:",score
        
    print 'result score : %f' % all_score


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
