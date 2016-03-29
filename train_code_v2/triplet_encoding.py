import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U


def build(P , word_size , first_hidden_size , encoding_size) :
    """
    create entity and relation encoding
    """
    P["W_word_left_input"]  = U.initial_weights(2*word_size , first_hidden_size)
    P["W_word_right_input"] = U.initial_weights(2*word_size , first_hidden_size)
    P["W_encoding_output"]  = U.initial_weights(2*first_hidden_size , encoding_size)

    def batched_triplet_encoding(e_left , relation , e_right) :
        left_input  = T.concatenate([e_left  , relation] , axis=1) #batched version
        right_input = T.concatenate([relation , e_right] , axis=1) #batched version

        left_hidden  = T.tanh(T.dot(left_input  , P["W_word_left_input"]))
        right_hidden = T.tanh(T.dot(right_input , P["W_word_right_input"]))

        all_hidden = T.concatenate([left_hidden , right_hidden] , axis = 1) #batched version
        encoding = T.tanh(T.dot(all_hidden , P["W_encoding_output"]))

        return encoding
    def vector_triplet_encoding(e_left , relation , e_right) :
        left_input  = T.concatenate([e_left  , relation] , axis=0) #batched version
        right_input = T.concatenate([relation , e_right] , axis=0) #batched version

        left_hidden  = T.tanh(T.dot(left_input  , P["W_word_left_input"]))
        right_hidden = T.tanh(T.dot(right_input , P["W_word_right_input"]))

        all_hidden = T.concatenate([left_hidden , right_hidden] , axis = 0) #batched version
        encoding = T.tanh(T.dot(all_hidden , P["W_encoding_output"]))

        return encoding
    return batched_triplet_encoding , vector_triplet_encoding