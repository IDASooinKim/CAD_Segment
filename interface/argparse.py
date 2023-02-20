# -- coding: utf-8 -*-  

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
"""

import argparse


def get_args():

    parser = argparse.ArgumentParser(description='Argment for train and inference')
    
    # settings for training hyper paramerters
    parser.add_argument('--epochs', '-e', dest='epochs',
                        metavar='E', type=int, default=1000, help='Number of Epochs')
    parser.add_argument('--learning_rate', '-l', dest='learning_rate', metavar='LR',
                        type=float, default=0.0001, help='Number of Learning Rate')
    parser.add_argument('--decay', '-dc', dest='decay',
                        metavar='D', type=float, default=0.9)
    parser.add_argument('--batch_size', '-b', dest='batch_size',
                        metavar='B', type=int, default=64, help='Number of Batch Size')

    parser.add_argument('--n_classes', '-c', dest='n_classes',
                        metavar='L', type=int, default=11, help='Size of Classify Tensor')
    parser.add_argument('--n_channels', '-n', dest='n_channels',
                        metavar='C', type=int, default=1, help='Number of Input Channels')
    
    parser.add_argument('--test_ratio', '-t', dest='test_ratio', metavar='T',
                        type=str, default=0.2, help='Ratio of test datasets')
    parser.add_argument('--seed', default=7740, type=int,
                        help='seed for initializing training. ')
    
    # arguments for the path of training dataset and log path
    parser.add_argument('--img_path', '-ip', dest='img_path',
            metavar='IP', type=str, default='')
    parser.add_argument('--save_model_path', '-mp', dest='save_model_path', metavar='SP', type=str,
                        default='./logs', help='Path for Saving Model')
    
    # argumerents for the path of inference 
    parser.add_argument('--save_infer_result', '-sr', dest='save_infer_result', metavar='SR', type=str,
                        default='./results', help='Path for Saving results'
    )
    parser.add_argument('--saved_model_name', '-smn', dest='saved_model_name', metavar='SM', type=str,
                        default='1900_loss.pt', help='Name of models for inference'
    )

    # arguments for the distributed data parallel training
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--dist_url', default='env://', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--world_size', default=4, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('--num_workers', default=4, type=int,
                    help='number of workers for data load')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
    
    return parser.parse_args()
