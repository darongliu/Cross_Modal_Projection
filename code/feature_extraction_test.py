import numpy as np

caffe_root = '../../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import os

"""
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
"""
caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

#customed by user
import pickle
train_path_file = '../metadata/train_path'
test_path_file  = '../metadata/test_path'
image_root      = '../images'
save_dir        = './test'

train = pickle.load(open(train_path_file,'rb'))
test  = pickle.load(open(test_path_file ,'rb'))


for layer in ['fc6','fc7']:
    result = []
    count = 0
    print 'extract layer', layer
    print 'extract train'
    for example in train :
        full_path = os.path.join(image_root,example)
        print 'extract', full_path  
        try :
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(full_path))
            out = net.forward()
            feat = np.array(net.blobs[layer].data[0])
        except :
            feat = None
        print feat
        result.append(feat)
        count = count + 1
        if count == 3 : 
            break
    pickle.dump(result,open(os.path.join(save_dir,'train_'+layer),'wb'))

    print 'extract test'
    result = []
    count = 0
    for example in test :
        full_path = os.path.join(image_root,example)
    
        try :    
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(full_path))
            out = net.forward()
            feat = np.array(net.blobs[layer].data[0])
        except :
            feat = None
        print feat
        count = count + 1
        if count == 3 :
            break
        result.append(feat)
    pickle.dump(result,open(os.path.join(save_dir,'test_'+layer),'wb'))
