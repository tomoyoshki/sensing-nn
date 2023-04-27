import os
import warnings
import numpy as np
from tqdm import tqdm
import torch
import seaborn as sns

warnings.simplefilter("ignore", UserWarning)

import torch.nn as nn

# utils
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.model_selection import init_backbone_model
from train_utils.knn import compute_knn, extract_sample_features

# Sklearn
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


def eval_tsne(args):
    """The main function for KNN test."""
    # Init data loaders
    test_dataloader = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

    # Only import augmenter after parse arguments so the device is correct
    from data_augmenter.Augmenter import Augmenter

    augmenter = Augmenter(args)
    augmenter.to(args.device)
    args.augmenter = augmenter

    # Init the classifier model
    classifier = init_backbone_model(args)
    pretrain_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_latest.pt")
    classifier = load_model_weight(classifier, pretrain_weight, load_class_layer=False)
    args.classifier = classifier

    # eval_backbone_tsne(args, classifier, augmenter, test_dataloader)
    eval_backbone_tsne_mod(args, classifier, augmenter, test_dataloader)


def eval_backbone_tsne(args, classifier, augmenter, dataloader):
    classifier.eval()

    sample_embeddings = []
    labels = []
    with torch.no_grad():
        for time_loc_inputs, label, index in tqdm(dataloader, total=len(dataloader)):
            """Move idx to target device, save label"""
            index = index.to(args.device)
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 else label
            labels.append(label.cpu().numpy())

            """Eval KNN estimator."""
            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            feat = extract_sample_features(
                args, classifier, aug_freq_loc_inputs, proj_head=args.learn_framework == "CMCV2"
            )
            sample_embeddings.append(feat.detach().cpu().numpy())

    # knn predictions
    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    z = tsne.fit_transform(sample_embeddings)
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    dataset = args.dataset
    dataset = dataset.replace("Parkland", "MOD")
    dataset = dataset.replace("RealWorld_HAR", "RealWorld-HAR")
    model = args.model
    model = model.replace("TransformerV4", "SW-T")
    framework = args.learn_framework
    framework = framework.replace("CMCV2", "FOCAL")
    res_dir = "../result/tsne_res"

    scatter_plot = sns.scatterplot(
        x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 10), legend=None, data=df
    )
    if os.path.exists(res_dir) == False:
        os.mkdir(res_dir)
    if os.path.exists(os.path.join(res_dir, dataset)) == False:
        os.mkdir(os.path.join(res_dir, dataset))
    # scatter_plot.set(title=f"{dataset} {model} {framework} data T-SNE projection")
    scatter_plot.set(title="")
    scatter_plot.set(xticklabels=[])
    scatter_plot.set(xlabel=None)
    scatter_plot.set(yticklabels=[])
    scatter_plot.set(ylabel=None)
    # scatter_plot.margins(x=0)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(res_dir, dataset), f"{model}_{framework}.pdf"))


def eval_backbone_tsne_mod(args, classifier, augmenter, dataloader):
    classifier.eval()

    sample_embeddings = {mod: [] for mod in args.dataset_config["modality_names"]}
    labels = []
    with torch.no_grad():
        for time_loc_inputs, label, index in tqdm(dataloader, total=len(dataloader)):
            """Move idx to target device, save label"""
            index = index.to(args.device)
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 else label
            labels.append(label.cpu().numpy())

            """Eval KNN estimator."""
            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            mod_feat = classifier(aug_freq_loc_inputs, class_head=False)
            for mod in args.dataset_config["modality_names"]:
                sample_embeddings[mod].append(mod_feat[mod].detach().cpu().numpy())

    # knn predictions
    labels = np.concatenate(labels)
    markers_dict = ["x", "v", "*", "+", "D", "2", "o"]

    # df = pd.DataFrame({"y": [], "comp-1": [], "comp-2": [], "marker": [], "mod": []})
    df = pd.DataFrame()
    for i, mod in enumerate(args.dataset_config["modality_names"]):
        features = np.concatenate(sample_embeddings[mod])
        tsne = TSNE(n_components=2, verbose=0, random_state=123)
        z = tsne.fit_transform(features)
        mod_df = pd.DataFrame()
        mod_df["y"] = labels
        mod_df["comp-1"] = z[:, 0]
        mod_df["comp-2"] = z[:, 1]
        mod_df["marker"] = markers_dict[i]
        mod_df["mod"] = mod
        df = df.append(mod_df, ignore_index=True)
    dataset = args.dataset
    dataset = dataset.replace("Parkland", "MOD")
    dataset = dataset.replace("RealWorld_HAR", "RealWorld-HAR")
    model = args.model
    model = model.replace("TransformerV4", "SW-T")
    framework = args.learn_framework
    framework = framework.replace("CMCV2", "FOCAL")
    res_dir = "../result/tsne_res"
    scatter_plot = sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", 10),
        legend=None,
        style="marker",
        data=df,
    )
    if os.path.exists(res_dir) == False:
        os.mkdir(res_dir)
    if os.path.exists(os.path.join(res_dir, dataset)) == False:
        os.mkdir(os.path.join(res_dir, dataset))
    # scatter_plot.set(title=f"{dataset} {model} {framework} data T-SNE projection")
    scatter_plot.set(title="")
    scatter_plot.set(xticklabels=[])
    scatter_plot.set(xlabel=None)
    scatter_plot.set(yticklabels=[])
    scatter_plot.set(ylabel=None)
    # scatter_plot.margins(x=0)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(res_dir, dataset), f"{model}_{framework}.pdf"))


def main_eval():
    """The main function of training"""
    args = parse_test_params()
    eval_tsne(args)


if __name__ == "__main__":
    main_eval()
