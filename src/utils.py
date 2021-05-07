import numpy as np
import torch
from distutils.version import LooseVersion
import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    center = kernel_size / 2 - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class cross_entropy2d_loss(nn.Module):
    def __init__(self):
        super(cross_entropy2d_loss, self).__init__()

    def forward(self, input, target, weight=None, size_average=True):
        loss = self.cal_loss(input, target, weight, size_average)
        return loss

    def cal_loss(self, input, target, weight, size_average):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        if LooseVersion(torch.__version__) < LooseVersion('0.3'):
            # ==0.2.X
            log_p = F.log_softmax(input)
        else:
            # >=0.3
            log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
        if size_average:
            loss /= mask.data.sum()
        return loss


class TripletLossFunc(nn.Module):
    def __init__(self, t1, t2, beta):
        super(TripletLossFunc, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.beta = beta
        return

    def forward(self, anchor, positive, negative):
        matched = torch.pow(F.pairwise_distance(anchor, positive), 2)
        mismatched = torch.pow(F.pairwise_distance(anchor, negative), 2)
        part_1 = torch.clamp(matched - mismatched, min=self.t1)
        part_2 = torch.clamp(matched, min=self.t2)
        dist_hinge = part_1 + self.beta * part_2
        loss = torch.mean(dist_hinge)
        return loss


if __name__ == "__main__":
    a = TripletLossFunc()
    loss = a(anchor=1, positive=2, negative=3)