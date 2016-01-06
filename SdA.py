#!/usr/bin/env python
""" Dimensionality reduction by Stacked denoising Autoencoders
"""
import argparse
import logging

import numpy
import numpy as np
import theano
import theano.tensor as T

from sda.SdA import SdA

logging.basicConfig(level=logging.INFO)


def SdAWrapper(X, batch_size = 128, layers = [512, 64], corruption_levels = [0.3, 0.3], pretrain_epochs = 100, pretrain_lr = 0.001):
    X = theano.shared(np.asarray(X, dtype = theano.config.floatX), borrow = True)
    n_samples, n_vars = X.get_value(borrow=True).shape
    n_train_batches = n_samples / batch_size
    numpy_rng = numpy.random.RandomState(23432)
    ###############
    # BUILD MODEL #
    ###############
    logging.info('Building model')
    logging.info(str(n_vars) + ' -> ' + ' -> '.join(map(str, layers)))

    sda = SdA(numpy_rng=numpy_rng, n_ins=n_vars,
              hidden_layers_sizes=layers)

    logging.info('Compiling training functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=X,
                                                batch_size=batch_size)

    #####################
    # TRAINING MODEL #
    #####################
    logging.info('Training model')
    for i in xrange(sda.n_layers):
        for epoch in xrange(pretrain_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))
            logging.info('Training layer {}, epoch {}, cost {}'.format(
                i, epoch, numpy.mean(c)))

    return sda

def main(args):
    logging.info('Loading data')
    
    X = np.load('VOC2007_feats.npy')

    




    y = sda.get_lowest_hidden_values(X)
    get_y = theano.function([], y)
    y_val = get_y()
    np.save('VOC2007_feats_encode.npy', y_val)
    # print_array(y_val, index=index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('input')
    parser.add_argument('-l', nargs='+', type=int, default=[2],
                        help='List of hidden layer sizes, last size'
                        ' will be the final dimension of output')
    args = parser.parse_args()
    
    main(args)
