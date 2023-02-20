#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun train.py --multiprocessing_distributed --rank 0 --batch_size 32 --epochs 2000 --save_model_path './logs' --world_size 1 --img_path '/data/01_CAD/solideos_4ch_5ch_all_dataset_v4.0.hdf5' --n_channels 1 --n_classes 7
