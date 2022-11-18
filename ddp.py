import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import os
import torch.distributed as dist

# for reproducibility
import random
import numpy as np
import torch.backends.cudnn as cudnn

# ABOUT DDP
# for model loading in ddp mode
from torch.nn.parallel import DistributedDataParallel as DDP
# for data loading in ddp mode
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group



def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def train_net(net,
              device,
              start: int = 0,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    

    if DDP_ON: # modify the net's attributes when using ddp
        net.n_channels = net.module.n_channels
        net.n_classes  = net.module.n_classes

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=WORLD_SIZE*4, pin_memory=True) # batchsize is for a single process(GPU)

    if DDP_ON:
        train_sampler = DistributedSampler(train_set)
        train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    
    
    # no need for distributed sampler for val
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    # (Initialize logging)
    if LOCAL_RANK == 0:
        experiment = wandb.init(project='U-Net-DDP', resume='allow', anonymous='must')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))
            
        logging.info(f'''Starting training:
                Epochs:          {epochs}
                Start from:      {start}
                Batch size:      {batch_size}
                Learning rate:   {learning_rate}
                Training size:   {n_train}
                Validation size: {n_val}
                Checkpoints:     {save_checkpoint}
                Device:          {device.type}
                Images scaling:  {img_scale}
                Mixed Precision: {amp}
            ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    criterion = nn.CrossEntropyLoss() 
    
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    # 5. Begin training
    for epoch in range(start, start+epochs):
        if LOCAL_RANK == 0:
            print('lr: ', optimizer.param_groups[0]['lr']) 
        
        net.train()
        epoch_loss = 0

        # To avoid duplicated data sent to multi-gpu
        train_loader.sampler.set_epoch(epoch)

        disable = False if LOCAL_RANK == 0 else True

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs+start}', unit='img', disable=disable) as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                    
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                if LOCAL_RANK == 0:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device, disable_log = disable)

                        if LOCAL_RANK == 0:
                            logging.info('Validation Dice score: {}'.format(val_score))
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
        scheduler.step()
        if save_checkpoint and LOCAL_RANK == 0 and (epoch % args.save_every == 0):
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.module.state_dict(), str(dir_checkpoint / 'DDP_checkpoint_epoch{}.pth'.format(epoch)))
            
            logging.info(f'Checkpoint {epoch} saved!')


##################################### arguments ###########################################
parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                    help='Learning rate', dest='lr')
parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                    help='Percent of the data that is used as validation (0-100)')
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
parser.add_argument('--exp_name', type=str, default='wxg_exp')
parser.add_argument('--ddp_mode', action='store_true')
parser.add_argument('--save_every', type=int, default=5)
parser.add_argument('--start_from', type=int, default=0)




args = parser.parse_args()

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

DDP_ON = True if args.ddp_mode else False

#########################################################################################

if DDP_ON:
    init_process_group(backend="nccl")
    LOCAL_RANK = device_id = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = torch.cuda.device_count()

    device = torch.device('cuda', device_id) # note that device_id is an integer but device is a datetype.
    print(f"Start running basic DDP on rank {LOCAL_RANK}.")
    logging.info(f'Using device {device_id}')


if __name__ == '__main__':
    #!highly recommended]
    # ref: pytorch org ddp tutorial 
    # 1. https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html#multi-gpu-training-with-ddp
    # 2. https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
    
    init_seeds(0)
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    
    if LOCAL_RANK == 0:
        print(f'Network:\n'
            f'\t{net.n_channels} input channels\n'
            f'\t{net.n_classes} output channels (classes)\n'
            f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        # ref: https://blog.csdn.net/hustwayne/article/details/120324639  use method 2 with module
        # net.load_state_dict(torch.load(args.load, map_location=device))
        net.load_state_dict({k.replace('module.', ''): v for k, v in                 
                       torch.load(args.load, map_location=device).items()})

        logging.info(f'Model loaded from {args.load}')


    torch.cuda.set_device(LOCAL_RANK)
    net.to(device=device)
    # wrap our model with ddp
    net = DDP(net, device_ids = [device_id], output_device=device_id)

    try:
        train_net(net=net,
                  start=args.start_from,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.module.state_dict(), 'INTERRUPTED_DDP.pth')
        logging.info('Saved interrupt')
        raise
    destroy_process_group()