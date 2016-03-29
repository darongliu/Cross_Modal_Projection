import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U

def build(P, input_size, proj_size) :
    P["image_projection matrix"] = U.initial_weights(input_size, proj_size) #issue: initial method

    def image_project(x) :
        #projection
        proj_result = T.dot(x,P["image_projection matrix"])
        return proj_result #whether normalize or not
        
    return image_project

