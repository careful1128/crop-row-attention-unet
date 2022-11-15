#!/bin/bash
export OMP_NUM_THREADS=1
torchrun --nproc_per_node 4 \
            train.py \
            --epochs 50 \
            --learning-rate 1e-5 \
            --batch-size 4 \
            --scale 1 \
            --validation 10 \
            --amp \
            --ddp_mode \
            --start_from 0 \
            --exp_name wxg \
            # --load /home/shenlan07/Pytorch-UNet-master2/checkpoints/DDP_checkpoint_epoch48.p