#!/bin/sh

# Folders definition
maestro_path='data/maestro-v3.0.0'
maestro_cache_path='data_cache'
exp_dir='exp/exp_v1'

# phase1 training parameters
phase_1_batch_size=6
phase_1_n_epochs=7
phase_1_learning_rate=0.001

# phase2 training parameters
phase_2_batch_size=3
phase_2_n_epochs=3
phase_2_learning_rate=0.00001

# phase3 training parameters
phase_3_batch_size=6
phase_3_n_epochs=10
phase_3_learning_rate=0.001


python3 train.py \
	--batch_size $phase_1_batch_size \
	--epochs $phase_1_n_epochs \
	--lr $phase_1_learning_rate \
	--phase 1 \
	$maestro_path \
	$maestro_cache_path \
	$exp_dir
