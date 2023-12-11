import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import canny
    

class FourDomainLoss(nn.Module):
    """Frequency Domain Loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        y_hat = torch.fft.rfft2(y_hat)
        y = torch.fft.rfft2(y)

        return F.l1_loss(y_hat, y)
    

class EdgeLoss(nn.Module):
    """Edge Details Loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        _, y_hat_edges = canny(y_hat)
        _, y_edges = canny(y)
        return F.l1_loss(y_hat_edges, y_edges)