import torch
from torch import Tensor
import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import torch.nn as nn

def soft_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    criterion = nn.CrossEntropyLoss()
    target = target.cpu().numpy()
    for i in range(target.shape[0]):
        k = 5
        kernel = np.ones((k, k), np.uint8)
        img = target[i]*255
        x = cv2.dilate(img.astype(np.uint8), kernel)
        x = distance_transform_edt(x)
        x = x/x.max() * 255
        mask = np.array(img>x)
        img = img * mask
        x = x * (1-mask)
        x = x + img
        x = x/x.max()
        # x = (x)/(x.max()+1e-6)

        cv2.imwrite("soft_mask.png", x*255)
        target[i] = x
    target = torch.from_numpy(target).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return criterion(input, target)


def soft_loss2(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    loss = 0
    for i in range(target.shape[0]):
        x = target[i]*255
        x = distance_transform_edt(255-x.cpu())
        x = torch.Tensor(x/x.max()).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        pred = input[i]
        x = x * pred
        for cls in range(x.shape[0]):
            loss += torch.exp(torch.mean(x[cls]))-1
    
    loss = loss / target.shape[0]
    cv2.imwrite("soft_mask.png", x[1].detach().cpu().numpy()*255)
    # target = torch.from_numpy(target).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # return loss
    return torch.sqrt(loss)
