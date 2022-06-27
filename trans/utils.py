from random import random
import torch
from torch import nn

# original: https://github.com/VITA-Group/TransGAN/blob/51b00d9ebdbcbeb42f4be2181fed394219f10e73/models_search/ViT_helper.py#L22

def drop_path(x, drop_prob: float = 0, training: bool = False):
    """
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    author use 'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    
    # work with diff dim tensors, not just 2D ConvNets
    ## with diff dim tensors, use ","
    shape = (x.shape[0],) + (1,) * (x.dim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)