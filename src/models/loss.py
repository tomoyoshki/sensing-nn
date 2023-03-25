"""
referebce: @https://github.com/facebookresearch/dino
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from general_utils.alias_multimodal import AliasMethod


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

    def forward(self, z_i, z_j, idx=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        # Calculate similarity
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

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

        # Calculate similarity
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
        loss = self.criterion(logits, labels) / N

        return loss

    def forward(self, seismic_features, audio_features, index):
        loss = self.forward_similiarity(seismic_features, audio_features)

        return loss


class CMCV2Loss(nn.Module):
    def __init__(self, args):
        super(CMCV2Loss, self).__init__()
        self.args = args
        self.config = args.dataset_config["CMCV2"]
        self.modalities = args.dataset_config["modality_names"]
        self.temperature = args.dataset_config[args.learn_framework]["temperature"]
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.orthonal_loss_f = nn.CosineEmbeddingLoss(reduction="mean")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward_contrastive_loss(self, mod1_feature, mod2_feature):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        batch_size = mod1_feature.shape[0]
        N = 2 * batch_size

        # Calculate similarity
        z = torch.cat((mod1_feature, mod2_feature), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim = sim.reshape(sim.shape[0], sim.shape[1])
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        # Compute loss
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss

    def forward_orthogonality_loss(self, embeddings1, embeddings2):
        """
        Compute the orthogonality loss for the modality features. No cross-sample operation is involved.
        input shape: [b. dim]
        """
        batch = embeddings1.shape[0]
        orthogonal_loss = self.orthonal_loss_f(
            embeddings1,
            embeddings2,
            target=torch.ones(batch).to(embeddings1.device),
        )

        return orthogonal_loss

    def forward(self, mod_features1, mod_features2, index=None):
        """
        loss = shared contrastive loss + mod contrastive loss + orthogonality loss
        """
        loss = 0

        # split features into "shared" space and "private" space
        split_mod_features1, split_mod_features2 = {}, {}
        for mod in self.modalities:
            dim = mod_features1[mod].shape[1]
            split_mod_features1[mod] = {
                "shared": mod_features1[mod][:, 0 : (dim // 2)],
                "private": mod_features1[mod][:, (dim // 2) :],
            }

            dim = mod_features2[mod].shape[1]
            split_mod_features2[mod] = {
                "shared": mod_features2[mod][:, 0 : (dim // 2)],
                "private": mod_features2[mod][:, (dim // 2) :],
            }

        # shared space contrastive loss
        for split_mod_features in [split_mod_features1, split_mod_features2]:
            for i, mod1 in enumerate(self.modalities):
                for mod2 in self.modalities[i:]:
                    loss += self.forward_contrastive_loss(
                        split_mod_features[mod1]["shared"],
                        split_mod_features[mod2]["shared"],
                    )

        # private space contrastive loss
        for mod in self.modalities:
            loss += self.forward_contrastive_loss(
                split_mod_features1[mod]["private"],
                split_mod_features2[mod]["private"],
            )

        # orthogonality loss
        for split_mod_features in [split_mod_features1, split_mod_features2]:
            for i, mod in enumerate(self.modalities):
                # orthognoality between shared and private space
                loss += self.forward_orthogonality_loss(
                    split_mod_features[mod]["shared"],
                    split_mod_features[mod]["private"],
                )

                # orthogonality between modalities
                for mod2 in self.modalities[i:]:
                    loss += self.forward_orthogonality_loss(
                        split_mod_features[mod]["private"],
                        split_mod_features[mod2]["private"],
                    )

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

    def calc_loss(self, mod_features):
        """
        mod_features: [mod, batch, channel]
        """
        mods, batch, channel = mod_features.shape

        # Positive Pairs
        sample_features = mod_features.permute(1, 0, 2)
        sim = torch.matmul(sample_features, sample_features.permute(0, 2, 1))
        sim = torch.ones((batch, mods, mods)).to(self.args.device).float() - sim
        sim = torch.exp(sim / self.temperature)
        pos_error = torch.mean(sim)

        # Negative Pairs
        sim = torch.matmul(mod_features, mod_features.permute(0, 2, 1))
        sim = torch.exp(sim / self.temperature)
        tri_mask = torch.ones([batch, batch]).to(self.args.device).bool()
        tri_mask.fill_diagonal_(False).unsqueeze(0).repeat(mods, 1, 1)
        off_diag_sim = torch.masked_select(sim, tri_mask).reshape([mods, batch, batch - 1])
        neg_error = torch.sum(torch.mean(off_diag_sim, dim=[1, 2]), dim=0)

        loss = self.scale_loss * torch.mean(pos_error) + self.lambd * torch.mean(neg_error)

        return loss

    def forward(self, mod_features):
        features = torch.stack([mod_features[mod] for mod in self.modalities], dim=0)
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

                loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                total_loss += loss

        return total_loss
