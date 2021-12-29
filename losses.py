import torch
from utils import label_to_one_hot
import torch.nn.functional as F

def dice_loss(pred, target):
    """
    Calculates unweighted dice loss
    
    Parameters
    ----------
    pred : BxCxHxW one-hot predicted probability tensor
    target : BxCxHxW one-hot ground truth tensor

    Returns
    -------
    loss : scalar tensor

    """
    smooth = 1.0

    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def dice_loss_logits(pred, target):
    """
    Dice loss with pred being logits, see dice_loss

    """
    pred_sm = F.softmax(pred, dim=1)
    return dice_loss(pred_sm, target)

def iou(pred, target):
    """
    Calculates intersection-over-union
    
    Parameters
    ----------
    pred : BxCxHxW one-hot predicted probability tensor
    target : BxCxHxW one-hot ground truth tensor

    Returns
    -------
    iou : scalar tensor

    """
    epsilon = 1e-8
    pred_argmax = torch.argmax(pred, dim=1, keepdim=True)
    pred_discrete = label_to_one_hot(pred_argmax, pred.shape[1]).contiguous()
    iflat = pred_discrete.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum() - intersection
    return intersection/ (union + epsilon)

def iou_logits(pred, target):
    """ see iou """
    pred_sm = F.softmax(pred, dim=1)
    return iou(pred_sm, target)

# logits = torch.randn(4,10,32,32)
# labels = torch.ones(4,10,32,32)
# b = iou_logits(logits, labels)