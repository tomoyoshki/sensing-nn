"""
referebce: @https://github.com/facebookresearch/dino
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from general_utils.alias_multimodal import AliasMethod


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
        self.config = args.dataset_config[args.learn_framework]

        # hyperparameters
        self.temperature_s = self.config["temperature_student"]
        self.temperature_t = self.config["temperature_teacher"]
        self.center_momentum = self.config["center_momentum"]

        # centering vector
        self.register_buffer("center", torch.zeros(1, self.config["emb_dim"]))

    def forward(self, student_output, teacher_output, idx=None):
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
    def __init__(self, args):
        super(SimCLRLoss, self).__init__()
        self.batch_size = args.batch_size
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]

        self.mask = self.mask_correlated_samples(self.batch_size)
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

    def forward(self, z_i, z_j, idx=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
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


class MoCoLoss(nn.Module):
    """MoCo Loss function
    https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
    """

    def __init__(self, args):
        super(MoCoLoss, self).__init__()
        self.args = args
        self.T = args.dataset_config["MoCo"]["temperature"]

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # Einstein sum is more intuitive
        logits = torch.einsum("nc,mc->nm", [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).to(self.args.device)

        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, q, k, idx=None):
        q1, q2 = q
        k1, k2 = k
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        return loss


class CMCLoss(nn.Module):
    def __init__(self, args):
        super(CMCLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["CMC"]
        self.batch_size = args.batch_size
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]

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

    def forward_similiarity(self, z_i, z_j, idx=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim = sim.reshape(sim.shape[0], sim.shape[1])
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

    def forward(self, seismic_features, audio_features, index):
        loss = self.forward_similiarity(seismic_features, audio_features)

        return loss


class CosmoLoss(nn.Module):
    def __init__(self, args):
        """Based on SupConLoss from "Supervised Contrastive Loss" paper."""
        super(CosmoLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["Cosmo"]
        self.temperature = self.config["temperature"]
        self.contrast_mode = self.config["contrast_mode"]
        self.base_temperature = self.config["temperature"]

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, feat_dim].
        Returns:
            A loss scalar.
        """
        device = self.args.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # change to [n_views*bsz, dim]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.matmul(anchor_feature, contrast_feature.T)
        similarity_matrix = torch.div(similarity_matrix, self.temperature)

        # tile mask, [b * n_views, b * n_views]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask = mask.repeat(anchor_count, contrast_count)  # positive index

        # mask-out self-contrast cases in all diagonal positions; dig to 0, others to 1 (negative samples)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask  # positive samples except diagonal elements

        # compute log_prob, except the diagonal positions, [b * n_views, b * n_views]
        exp_logits = torch.exp(similarity_matrix) * logits_mask  # exp(z_i * z_a / T)

        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # sup_out

        # loss, [n_views, bsz]
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CocoaLoss(nn.Module):
    def __init__(self, args, idx=None):
        """Cocoa loss similar to CMC
        Reference: https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py
        Paper: https://dl.acm.org/doi/10.1145/3550316
        """
        super(CocoaLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["Cocoa"]
        self.modalities = args.dataset_config["modality_names"]

        self.temperature = self.config["temperature"]
        self.scale_loss = self.config["scale_loss"]
        self.lambd = self.config["lambd"]

    def calc_loss(self, features):
        dim_size, batch_size, mod_size = features.shape

        # Positive Pairs
        pos_error = []
        for i in range(batch_size):
            # similarity of each modality S_vw of
            sim = torch.matmul(features[:, i, :], features[:, i, :].T)
            # 1 - S
            sim = torch.subtract(torch.ones((dim_size, dim_size), dtype=torch.float32).to(self.args.device), sim)
            # (1 - s) / t
            # e ^ ((1 - s) / tau)
            sim = torch.exp(sim / self.temperature)
            pos_error.append(torch.mean(sim))

        pos_error = torch.stack(pos_error)

        # Negative Pairs
        neg_error = 0
        for i in range(dim_size):
            # dot product
            sim = torch.matmul(features[i], features[i].T)
            sim = torch.exp(sim / self.temperature)
            tri_mask = np.ones(batch_size**2, dtype=np.bool).reshape(batch_size, batch_size)
            tri_mask[np.diag_indices(batch_size)] = False

            off_diag_sim = torch.masked_select(sim, torch.from_numpy(tri_mask).to(sim.device))
            off_diag_sim = torch.reshape(off_diag_sim, (batch_size, batch_size - 1))
            neg_error += torch.mean(off_diag_sim, dim=-1)

        cross_mod_loss = torch.multiply(torch.sum(pos_error), self.scale_loss)
        discriminator_loss = self.lambd * torch.sum(neg_error)

        return cross_mod_loss + discriminator_loss

    def forward(self, mod_features):
        features = []
        for mod in self.modalities:
            features.append(mod_features[mod])

        features = torch.stack(features)

        # normalize each features
        features_norm = F.normalize(features, dim=-1)

        loss = self.calc_loss(features_norm)
        return loss


class MAELoss(nn.Module):
    """MAE Loss function
    https://github.com/facebookresearch/mae/blob/main/models_mae.py
    """

    def __init__(self, args):
        super(MAELoss, self).__init__()
        self.args = args
        self.modalities = args.dataset_config["modality_names"]
        self.backbone_config = args.dataset_config[args.model]
        self.generative_config = args.dataset_config["MAE"]
        self.locations = args.dataset_config["location_names"]
        self.norm_pix_loss = True

    def patchify(self, imgs, patch_size, in_channel):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        ph, pw = patch_size

        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw

        x = imgs.reshape(shape=(imgs.shape[0], in_channel, h, ph, w, pw))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, ph * pw * in_channel)
        return x

    def forward(self, padded_x, decoded_x, masks):
        total_loss = 0
        for loc in self.locations:
            for mod in self.modalities:
                mask = masks[loc][mod]
                pred = decoded_x[loc][mod]
                
                if self.args.model == "TransformerV4":
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"]["shake"][mod]
                    target = self.patchify(
                        padded_x[loc][mod],
                        self.backbone_config["patch_size"]["freq"][mod],
                        in_channel,
                    )
                else:
                    patch_size = self.generative_config["patch_size"][mod]
                    patch_area = patch_size[0] * patch_size[1]
                    in_channel = self.args.dataset_config["loc_mod_in_freq_channels"]["shake"][mod]
                    target = self.patchify(
                        padded_x[loc][mod],
                        patch_size,
                        in_channel,
                    )
                    
                    pred = self.patchify(
                        pred,
                        patch_size,
                        in_channel,
                    )
                    mask = mask.reshape((pred.shape[0], -1))
                

                if self.norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / (var + 1.0e-6) ** 0.5

                loss = (pred - target) ** 2
                loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                
                # print("Loss shape: ", loss.shape)
                loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                total_loss += loss

        return total_loss
