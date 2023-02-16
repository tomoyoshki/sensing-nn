import torch
import logging
import numpy as np
from tqdm import tqdm

# utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from train_utils.knn import extract_sample_features


def eval_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs, idx):
    """Eval the contrastive loss for one batch."""
    if args.contrastive_framework == "CMC":
        aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
        feature1, feature2 = default_model(aug_freq_loc_inputs_1)
        loss = loss_func(feature1, feature2, idx)
    elif args.contrastive_framework == "Cosmo":
        aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
        rand_fused_features = default_model(aug_freq_loc_inputs_1)
        loss = loss_func(rand_fused_features)
    else:
        aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
        aug_freq_loc_inputs_2 = augmenter.forward("random", time_loc_inputs)
        feature1, feature2 = default_model(aug_freq_loc_inputs_1, aug_freq_loc_inputs_2)
        loss = loss_func(feature1, feature2, idx)

    return loss


def eval_predictive_loss(args, default_model, augmenter, loss_func, time_loc_inputs):
    """Eval the predictive loss for one batch."""
    if args.predictive_framework == "MTSS":
        aug_freq_loc_inputs, pretrain_labels = augmenter.forward("random", time_loc_inputs, return_aug_id=True)
        pretrain_predictions = default_model(aug_freq_loc_inputs)
    elif args.predictive_framework == "ModPred":
        aug_freq_loc_inputs, pretrain_labels = augmenter.forward("random", time_loc_inputs, return_aug_mods=True)
        pretrain_predictions = default_model(aug_freq_loc_inputs)
    else:
        raise NotImplementedError(f"Predictive framwork {args.predictive_framework} yet implemented")

    loss = loss_func(pretrain_predictions, pretrain_labels)

    return loss


def eval_task_metrics(args, labels, predictions):
    """Evaluate the downstream task metrics."""
    acc_scores = []
    f1_scores = []

    # different acc and f1-score definitions for single-class and multi-class problems
    if args.multi_class:
        for i, class_name in enumerate(args.dataset_config["class_names"]):
            acc = accuracy_score(labels[:, i], predictions[:, i])
            f1 = f1_score(labels[:, i], predictions[:, i], zero_division=1)

            if acc <= 1 and f1 <= 1:
                acc_scores.append(acc)
                f1_scores.append(f1)

            mean_acc = np.mean(acc_scores)
            mean_f1 = np.mean(f1_scores)
            conf_matrix = []
    else:
        mean_acc = accuracy_score(labels, predictions)
        mean_f1 = f1_score(labels, predictions, average="macro", zero_division=1)
        conf_matrix = confusion_matrix(labels, predictions)

    return mean_acc, mean_f1, conf_matrix


def eval_supervised_model(args, classifier, augmenter, dataloader, loss_func):
    """Evaluate the performance on the given dataloader.

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
    """
    # set both the classifier and augmenter to eval mode
    classifier.eval()

    # iterate over all batches
    num_batches = len(dataloader)
    classifier_loss_list = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for i, (time_loc_inputs, labels, index) in tqdm(enumerate(dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(freq_loc_inputs)
            classifier_loss_list.append(loss_func(logits, labels).item())

            if args.multi_class:
                predictions = (logits > 0.5).float()
            else:
                predictions = logits.argmax(dim=1, keepdim=False)
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1, keepdim=False)

            # for future computation of acc or F1 score
            saved_predictions = predictions.cpu().numpy()
            saved_labels = labels.cpu().numpy()
            all_predictions.append(saved_predictions)
            all_labels.append(saved_labels)

    # calculate mean loss
    mean_classifier_loss = np.mean(classifier_loss_list)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # calculate the classification metrics
    mean_acc, mean_f1, conf_matrix = eval_task_metrics(args, all_labels, all_predictions)

    return mean_classifier_loss, mean_acc, mean_f1, conf_matrix


def eval_pretrained_model(args, default_model, estimator, augmenter, data_loader, loss_func):
    """Evaluate the performance with KNN estimator."""
    default_model.eval()
    if args.train_mode == "contrastive" and args.contrastive_framework in {"CMC"}:
        loss_func.contrast.eval()

    sample_embeddings = []
    labels = []
    loss_list = []
    with torch.no_grad():
        for time_loc_inputs, label, index in tqdm(data_loader, total=len(data_loader)):
            """Move idx to target device, save label"""
            index = index.to(args.device)
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 else label
            labels.append(label.cpu().numpy())

            """Eval pretrain loss."""
            if args.train_mode == "contrastive":
                loss = eval_contrastive_loss(args, default_model, augmenter, loss_func, time_loc_inputs, index).item()
            elif args.train_mode == "predictive":
                loss = eval_predictive_loss(args, default_model, augmenter, loss_func, time_loc_inputs).item()
            loss_list.append(loss)

            """Eval KNN estimator."""
            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            feat = extract_sample_features(args, default_model.backbone, aug_freq_loc_inputs)
            sample_embeddings.append(feat.detach().cpu().numpy())

    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)
    predictions = torch.Tensor(estimator.predict(sample_embeddings))
    predictions = predictions.argmax(dim=1, keepdim=False) if predictions.dim() > 1 else predictions

    # compute metrics
    mean_loss = np.mean(loss_list)
    mean_acc, mean_f1, conf_matrix = eval_task_metrics(args, labels, predictions)

    return mean_loss, mean_acc, mean_f1, conf_matrix


def val_and_logging(
    args,
    epoch,
    tb_writer,
    model,
    augmenter,
    val_loader,
    test_loader,
    loss_func,
    train_loss,
    tensorboard_logging=True,
    estimator=None,
):
    """Validation and logging function.

    Args:
        tb_writer (_type_): _description_
        classifier (_type_): _description_
        miss_simulator (_type_): _description_
        val_dataloader (_type_): _description_
        classifier_dataloader (_type_): _description_
        classifier_loss_func (_type_): _description_
    """
    if args.train_mode in {"contrastive", "predictive"} and args.stage == "pretrain":
        logging.info(f"Train {args.train_mode} loss: {train_loss: .5f} \n")
    else:
        logging.info(f"Train classifier loss: {train_loss: .5f} \n")

    if estimator is None:
        """Supervised training or fine-tuning"""
        val_loss, val_f1, val_acc, val_conf_matrix = eval_supervised_model(
            args, model, augmenter, val_loader, loss_func
        )
        test_loss, test_f1, test_acc, test_conf_matrix = eval_supervised_model(
            args, model, augmenter, test_loader, loss_func
        )
    else:
        """Self-supervised pre-training"""
        val_loss, val_f1, val_acc, val_conf_matrix = eval_pretrained_model(
            args, model, estimator, augmenter, val_loader, loss_func
        )
        test_loss, test_f1, test_acc, test_conf_matrix = eval_pretrained_model(
            args, model, estimator, augmenter, test_loader, loss_func
        )
    logging.info(f"Val loss: {val_loss: .5f}")
    logging.info(f"Val acc: {val_acc: .5f}, val f1: {val_f1: .5f}")
    logging.info(f"Val confusion matrix:\n {val_conf_matrix} \n")
    logging.info(f"Test loss: {test_loss: .5f}")
    logging.info(f"Test acc: {test_acc: .5f}, test f1: {test_f1: .5f}")
    logging.info(f"Test confusion matrix:\n {test_conf_matrix} \n")

    # write tensorboard train log
    if tensorboard_logging:
        tb_writer.add_scalar("Validation/Val loss", val_loss, epoch)
        tb_writer.add_scalar("Validation/Val accuracy", val_acc, epoch)
        tb_writer.add_scalar("Validation/Val F1 score", val_f1, epoch)
        tb_writer.add_scalar("Evaluation/Test loss", test_loss, epoch)
        tb_writer.add_scalar("Evaluation/Test accuracy", test_acc, epoch)
        tb_writer.add_scalar("Evaluation/Test F1 score", test_f1, epoch)

    return val_acc, val_loss
