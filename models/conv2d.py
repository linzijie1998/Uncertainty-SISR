# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


class NormalGammaConv2D(nn.Module):
    def __init__(self, in_channels, n_task, **kwargs):
        super(NormalGammaConv2D, self).__init__()
        self.n_task = n_task
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=n_task * 4, **kwargs)

    def forward(self, x):
        x = self.conv(x)

        gamma, log_mu, log_alpha, log_beta = torch.split(x, self.n_task, dim=1)

        mu = F.softplus(log_mu)
        alpha = F.softplus(log_alpha) + 1
        beta = F.softplus(log_beta)

        return torch.cat([gamma, mu, alpha, beta], dim=1)


if __name__ == '__main__':
    x = torch.randn((8, 3, 80, 80))
    norm = nn.InstanceNorm2d(3)
    y = norm(x)
    print(y.size())
