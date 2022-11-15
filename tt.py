import os
import torch
import torch.distributed as dist
# import cv2

# Specify visible GPU to program
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2, 3, 4, 5, 6, 7]))

# DDP runs multi-process, and every process should obtain their local rank
# Store it as a global variable is recomended.
LOCAL_RANK = int(os.environ["LOCAL_RANK"]) 

# The device that we need to map our model to at the loading phase
DEVICE = torch.device("cuda", LOCAL_RANK)

# DDP backend initialization
torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group(backend='nccl')

# cv2.imread()
