import torch
from typing import List
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.losses import FocalLoss
from segmentation_models_pytorch.losses import SoftCrossEntropyLoss
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import CrossEntropyLoss

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"

class custom_loss(base.Loss):
    def __init__(self, batch_size, loss_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(mode='multiclass') 
        self.focal = FocalLoss(mode='multiclass')
        self.ce = CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, y_pr, y_gt, ft1=None):

        print("\ny_pred - ", y_pr.shape)
        print("y_gt before unsqueeze - ", y_gt.shape)

        print("y_pr - ", y_pr.shape)
        print("y_t - ", y_gt.shape)

        y_gt = y_gt.to(torch.int64)
        if y_gt.ndim == 4:
            y_gt = y_gt.squeeze(1)
        
        # y=self.dice(y_pr, y_gt)
        y = self.ce(y_pr, y_gt)
        f = self.focal(y_pr, y_gt)

        return (self.loss_weight)*y + (1-self.loss_weight)*f
    
        
