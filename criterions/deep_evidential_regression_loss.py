# -*- coding: utf-8 -*-

import torch
from torch import nn

import numpy as np


def nig_nll(y, gamma, upsilon, alpha, beta, reduce=True):
    """The Negative Logarithm of Model Evidence"""
    omega = 2 * beta * (1 + upsilon)

    part1 = 0.5 * torch.log(np.pi / upsilon)
    part2 = alpha * torch.log(omega)
    part3 = (alpha + 0.5) * torch.log(upsilon * (y - gamma) ** 2 + omega)
    part4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    nll = part1 - part2 + part3 + part4

    return torch.mean(nll) if reduce else nll


def nig_reg(y, gamma, upsilon, alpha, reduce=True):
    """Evidence Regularize"""
    error = torch.abs(y - gamma)
    evidence = 2 * upsilon + alpha
    reg = error * evidence
    return torch.mean(reg) if reduce else reg


class EvidentialLossSumOfSquares(nn.Module):
    def __init__(self, n_task=3, coefficient=0.01) -> None:
        super(EvidentialLossSumOfSquares, self).__init__()
        self.n_task = n_task
        self.coefficient = coefficient

        self.nll_loss = []
        self.reg_loss = []

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gamma, upsilon, alpha, beta = torch.split(inputs, self.n_task, dim=1)

        loss_nll = nig_nll(targets, gamma, upsilon, alpha, beta)
        loss_reg = nig_reg(targets, gamma, upsilon, alpha)

        # machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(inputs.device)
        # safe_nll = torch.max(machine_epsilon, loss_nll)

        self.nll_loss.append(loss_nll.item())
        self.reg_loss.append(loss_reg.item())

        return loss_nll + self.coefficient * loss_reg
