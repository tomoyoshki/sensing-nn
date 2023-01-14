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
    augmenter.eval()

    all_inputs = []
    labels = []
    for time_loc_inputs, y in data_loader_train:
        aug_freq_loc_inputs, _ = augmenter.forward(time_loc_inputs, y)
        all_inputs.append(classifier(aug_freq_loc_inputs).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    all_inputs = np.concatenate(all_inputs)
    labels = np.concatenate(labels)

    estimator = KNeighborsClassifier()
    estimator.fit(all_inputs, labels)

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
    augmenter.eval()

    embs_l = []
    time_loc_inputs_l = []
    labels = []

    classes = args.dataset_config["class_names"]

    for time_loc_inputs, y in data_loader:
        aug_freq_loc_inputs, y = augmenter.forward(time_loc_inputs, y)
        embs_l.append(classifier(aug_freq_loc_inputs).detach().cpu())
        # time_loc_inputs_l.append(time_loc_inputs["shake"]["seismic"])
        labels.extend([classes[i] for i in y.tolist()])

    embs = torch.cat(embs_l, dim=0)
    # time_loc_inputs_l = torch.cat(time_loc_inputs_l, dim=0)

    return embs, None, labels
