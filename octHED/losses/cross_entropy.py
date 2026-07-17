import torch
import torch.nn.functional as F

def cross_entropy_encoder_decoder(
        prediction: torch.tensor,                   
        labelef:    torch.tensor,
        std:        torch.tensor, 
        ada:        int 
        ):
    
    label = labelef.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()
    num_two = torch.sum((mask == 2).float()).float()
    assert (
        num_negative + num_positive + num_two
        == label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
    )
    assert num_two == 0
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    new_mask = mask * torch.exp(std * ada)
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=new_mask.detach(), reduction='sum'
    )

    return cost, mask

def weighted_cross_entropy_loss(
        preds: torch.tensor, 
        edges: torch.tensor
        ):
    
    """Calculate sum of weighted cross entropy loss."""
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7

    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float() 
    num_neg = c * h * w - num_pos  
    weight = torch.zeros_like(mask)
    weight[edges > 0.5] = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    losses = F.binary_cross_entropy(
        preds.float(), edges.float(), weight=weight, reduction='none'
    )
    loss = torch.sum(losses) / b
    return loss
