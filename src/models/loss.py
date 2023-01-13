"""
referebce: @https://github.com/facebookresearch/dino
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def cross_entropy_loss(logits, labels):
    """Compute the cross entropy loss between the logits and the labels.

    Args:
        logits (_type_): _description_
        labels (_type_): _description_
    """

    pass


class DINOLoss(nn.Module):
    def __init__(
        self,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        # student model hyper param
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # centering
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, 7))

    def forward(self, s1, s2, t1, t2):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        loss = (self.compute_loss(t1, s2) / 2) + (self.compute_loss(s1, t2) / 2)
        self.update_center(t1, t2)
        return loss

    def compute_loss(self, student, teacher):
        teacher = teacher.detach()
        student = F.softmax(student / self.student_temp, dim=1)
        teacher = F.softmax((teacher - self.center) / self.teacher_temp, dim=1)
        return -(teacher * torch.log(student)).sum(dim=1).mean()

    @torch.no_grad()
    def update_center(self, t1, t2):
        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * torch.cat([t1, t2]).mean(dim=0)
