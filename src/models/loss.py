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

    def __init__(self, args):
        super().__init__()
        self.config = args.dataset_config[args.contrastive_framework]

        # hyperparameters
        self.temperature_s = self.config["temperature_student"]
        self.temperature_t = self.config["temperature_teacher"]
        self.center_momentum = self.config["center_momentum"]

        # centering vector
        self.register_buffer("center", torch.zeros(1, self.config["emb_dim"]))

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
        student_out = student_output / self.temperature_s

        # teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / self.temperature_t, dim=-1)
        teacher_out = teacher_out.detach()

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    def H(self, feature_s, feature_t):
        """Sub-forward function used to calculate the loss

        Args:
            z_i (_type_): student predicted feature
            z_j (_type_): teacher predicted feature
        """
        feature_t = feature_t.detach()
        loss_s = F.softmax(feature_s / self.temperature_s, dim=1)
        loss_t = F.softmax((feature_t - self.center) / self.temperature_t, dim=1)
        return -(loss_t * torch.log(loss_s)).sum(dim=1).mean()

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
        # bc = cat([t1, t2]).mean(dim=0)
        batch_center = torch.cat((teacher_output[0], teacher_output[1])).mean(dim=0, keepdim=True)  # (1, out_dim)
        # C = (m * c) + (1-m) * (bc)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class SimCLRLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
