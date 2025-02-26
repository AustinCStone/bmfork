#!/bin/bash

# --input_dir is a directory of mp4 files
# --frames_dir is where the extracted png frames will be stored
# --dataset_dir is where the huggingface dataset (containing paths to frames) will be stored
# removing --overwrite will reuse the contents of --frames_dir when creating the dataset
# once your dataset is created, you can add its local path to base_miner/config.py for training

# create real frames dataset (example using SN34 validator cache)
python base_miner/datasets/create_video_dataset.py --input_dir ~/.cache/sn34/real/video \
       --frames_dir ~/.cache/sn34/real_frames \
       --dataset_dir ~/.cache/sn34/train_dataset/real_frames_dataset \
       --num_videos 500 \
       --frame_rate 5 \
       --max_frames 24 \
       --dataset_name real_frames \
       --overwrite 

# create synthetic frames dataset (example using SN34 validator cache)
python base_miner/datasets/create_video_dataset.py --input_dir ~/.cache/sn34/synthetic/t2v
       --frames_dir ~/.cache/sn34/synthetic_frames \
       --dataset_dir ~/.cache/sn34/train_dataset/synthetic_frames_dataset \
       --num_videos 500 \
       --frame_rate 5 \
       --max_frames 24 \
       --dataset_name synthetic_frames \
       --overwrite 
