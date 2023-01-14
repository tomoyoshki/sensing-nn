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
    """The loss function.
    We subclass the `nn.Module` becuase we want to create a buffer for the
    logits center of the teacher.
    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).
    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.
    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.
    """

    def __init__(self, out_dim=1024, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """Evaluate loss.
        Parameters
        ----------
        student_output, teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing
            logits. The length is equal to number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.
        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """

        s1, s2 = student_output
        t1, t2 = teacher_output

        loss = (self.H(t1, s2) / 2) + (self.H(s1, t2) / 2)
        self.update_center(teacher_output)

        return loss

    def H(self, s, t):
        t = t.detach()
        s = F.softmax(s / self.student_temp, dim=1)
        t = F.softmax((t - self.center) / self.teacher_temp, dim=1)
        return -(t * torch.log(s)).sum(dim=1).mean()

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.
        Compute the exponential moving average.
        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
