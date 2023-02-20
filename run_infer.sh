#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --multiprocessing_distributed --rank 0 --batch_size 1 --save_model_path './logs' --img_path '/data/01_CAD/solideos_4ch_5ch_all_dataset_v4.0.hdf5' --n_channels 1 --n_classes 7 --saved_model_name '1800_v4_ce.pt' --epochs 10000
