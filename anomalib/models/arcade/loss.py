"""Loss function for the Arcade Model Implementation."""


from __future__ import annotations

import torch
from torch import Tensor, nn
from kornia.losses import SSIMLoss
from torch import autograd

class GeneratorLoss(nn.Module):

    def __init__(self, wcritic=1, wssim=10) -> None: # 1 and 1
        super().__init__()

        self.loss_critic = 1
        self.loss_ssim = SSIMLoss(11, reduction='mean')

        self.wacritic = wcritic
        self.wssim = wssim

    def forward(
        self, images: Tensor, fake: Tensor, pred_fake: Tensor
    ) -> Tensor:
        loss_ssim1 = self.loss_ssim(images, fake)
        loss_critic = torch.sum(self.loss_critic * pred_fake)
        loss = loss_ssim1 * self.wssim - loss_critic * self.wacritic

        return loss


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for the Arcade model."""

    def __init__(self, wGP=100) -> None:
        super().__init__()
        self.wGP = wGP

    def forward(self, pred_real: Tensor, pred_fake: Tensor, interpolated, prob_interpolated) -> Tensor:
        # penalty
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(
                                    prob_interpolated.size()).cuda(),
                                create_graph=True, retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.wGP
        grad_penalty.backward()
        grad_penalty = torch.sum((pred_real - pred_fake)) + grad_penalty

        return grad_penalty