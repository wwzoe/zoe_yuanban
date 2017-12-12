# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from utils import DataHelper
import model
import evaluation

PATH_TO_TRAIN = './save/rsc15_train_full.txt'
PATH_TO_TEST = './save/rsc15_test.txt'

class Args():
    is_training = False
    batch_size = 5
    dropout_p_hidden=1
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    checkpoint_dir = './checkpoint'
    n_items = -1

def parseArgs():

    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--train', default=1,type=int)
    parser.add_argument('--layer', default=3, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    parser.add_argument('--save_dir', default='./save/', type=str)
    parser.add_argument('--data_path', default='./data/', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()

    args = Args()
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.data_path=command_line.data_path
    args.save_dir= command_line.save_dir

    data_loader = DataHelper(args, args.data_path, args.save_dir)
    data_loader.choose_data()

    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
    args.n_items = len(data['ItemId'].unique())

    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout

    print(args.n_epochs)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        if args.is_training==1:
            gru.fit(data)
        else:
            res = evaluation.evaluate_sessions_batch(gru, data, valid)
            print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
