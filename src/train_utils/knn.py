import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def compute_knn(args, classifier, augmenter, data_loader_train):
    """Get CLS embeddings and use KNN classifier on them.
    We load all embeddings in memory and use sklearn. Should
    be doable.
    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer whose head is just an identity
        mapping.
    data_loader_train, data_loader_val : torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any
        augmentations. Just casting to tensor and then normalizing.
    Returns
    -------
    val_accuracy : float
        Validation accuracy.
    """
    classifier.eval()

    sample_embeddings = []
    labels = []
    for time_loc_inputs, y, _ in data_loader_train:
        aug_freq_loc_inputs, _ = augmenter.forward("no", time_loc_inputs, y)
        if args.contrastive_framework == "CMC":
            knn_feature1, knn_feature2 = classifier(aug_freq_loc_inputs)
            feat = torch.cat((knn_feature1.detach(), knn_feature2.detach()), dim=1)
        else:
            feat = classifier(aug_freq_loc_inputs)
        sample_embeddings.append(feat.detach().cpu().numpy())

        labels.append(y.detach().cpu().numpy())

    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)

    estimator = KNeighborsClassifier()
    estimator.fit(sample_embeddings, labels)

    return estimator


def compute_embedding(args, classifier, augmenter, data_loader):
    """Compute CLS embedding and prepare for TensorBoard.
    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer. The head should be an identity mapping.
    data_loader : torch.utils.data.DataLoader
        Validation dataloader that does not apply any augmentations. Just
        casting to tensor and then normalizing.
    Returns
    -------
    embs : torch.Tensor
        Embeddings of shape `(n_samples, out_dim)`.
    imgs : torch.Tensor
        Images of shape `(n_samples, 3, height, width)`.
    labels : list
        List of strings representing the classes.
    """
    classifier.eval()

    embs_l = []
    all_labels = []
    classes = args.dataset_config["class_names"]

    for time_loc_inputs, labels, _ in data_loader:
        aug_freq_loc_inputs, y = augmenter.forward("no", time_loc_inputs, labels)
        if args.contrastive_framework == "CMC":
            seismic_feature, audio_feature = classifier(aug_freq_loc_inputs)
            features = torch.cat((seismic_feature.detach(), audio_feature.detach()), dim=1)
        else:
            features = classifier(aug_freq_loc_inputs)
        embs_l.append(features.detach().cpu())
        if labels.dim() > 1:
            labels = labels.argmax(dim=1, keepdim=False)
        all_labels.append([classes[i] for i in labels])

    embs = torch.cat(embs_l, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return embs, None, all_labels
