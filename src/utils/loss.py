import torch
from torch.nn import CrossEntropyLoss
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
from segmentation_models_pytorch.utils import base

class CustomLoss(base.Loss):
    def __init__(self, batch_size, loss_type="focal", loss_weight=0.5):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.loss_weight = loss_weight

        # Initialize the required losses
        self.dice = DiceLoss(mode="multiclass")
        self.focal = FocalLoss(mode="multiclass")
        self.ce = CrossEntropyLoss()

    def forward(self, y_pr, y_gt):
        y_gt = y_gt.to(torch.int64).squeeze(1)

        if self.loss_type == "focal":
            # Focal loss only
            return self.focal(y_pr, y_gt)

        elif self.loss_type == "ce+dice":
            # CrossEntropy + Dice loss
            ce_loss = self.ce(y_pr, y_gt)
            dice_loss = self.dice(y_pr, y_gt)
            return self.loss_weight * ce_loss + (1 - self.loss_weight) * dice_loss

        elif self.loss_type == "dice":
            # Dice loss only
            return self.dice(y_pr, y_gt)

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}. Choose from 'focal', 'ce+dice', 'dice'.")
