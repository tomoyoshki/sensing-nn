import torch
import logging
import numpy as np
from tqdm import tqdm

# utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


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
        for i, (time_loc_inputs, labels) in tqdm(enumerate(dataloader), total=num_batches):
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
    acc_scores = []
    f1_scores = []

    # different acc and f1-score definitions for single-class and multi-class problems
    if args.multi_class:
        for i, class_name in enumerate(args.dataset_config["class_names"]):
            acc = accuracy_score(all_labels[:, i], all_predictions[:, i])
            f1 = f1_score(all_labels[:, i], all_predictions[:, i], zero_division=1)

            if acc <= 1 and f1 <= 1:
                acc_scores.append(acc)
                f1_scores.append(f1)

            mean_acc = np.mean(acc_scores)
            mean_f1 = np.mean(f1_scores)
            conf_matrix = []
    else:
        mean_acc = accuracy_score(all_labels, all_predictions)
        mean_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=1)
        conf_matrix = confusion_matrix(all_labels, all_predictions)

    return mean_classifier_loss, mean_f1, mean_acc, conf_matrix


def eval_contrastive_model(model, estimator, augmenter, data_loader, loss_func):
    """Evaluate the performance with KNN estimator."""
    model.eval()

    sample_embeddings = []
    labels = []
    loss_list = []
    with torch.no_grad():
        for time_loc_inputs, label in tqdm(data_loader, total=len(data_loader)):
            """Eval contrastive loss."""
            aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
            aug_freq_loc_inputs_2 = augmenter.forward("random", time_loc_inputs)
            feature1, feature2 = model(aug_freq_loc_inputs_1, aug_freq_loc_inputs_2)

            """Eval KNN estimator."""
            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            sample_embeddings.append(model.backbone(aug_freq_loc_inputs, class_head=False).detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

            # forward pass
            loss = loss_func(feature1, feature2).item()
            loss_list.append(loss)

    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)
    predictions = estimator.predict(sample_embeddings)

    # compute metrics
    mean_acc = accuracy_score(labels, predictions)
    mean_f1 = f1_score(labels, predictions, average="macro", zero_division=1)
    conf_matrix = confusion_matrix(labels, predictions)
    mean_loss = np.mean(loss_list)

    return mean_loss, mean_f1, mean_acc, conf_matrix


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
    if args.train_mode in {"contrastive"} and args.stage == "pretrain":
        logging.info(f"Train contrastive loss: {train_loss: .5f} \n")
    else:
        logging.info(f"Train classifier loss: {train_loss: .5f} \n")

    if estimator is None:
        """Self-supervised pre-training"""
        val_loss, val_f1, val_acc, val_conf_matrix = eval_supervised_model(
            args, model, augmenter, val_loader, loss_func
        )
        test_loss, test_f1, test_acc, test_conf_matrix = eval_supervised_model(
            args, model, augmenter, test_loader, loss_func
        )
    else:
        """Supervised training or fine-tuning"""
        val_loss, val_f1, val_acc, val_conf_matrix = eval_contrastive_model(
            model, estimator, augmenter, val_loader, loss_func
        )
        test_loss, test_f1, test_acc, test_conf_matrix = eval_contrastive_model(
            model, estimator, augmenter, test_loader, loss_func
        )
    logging.info(f"Val loss: {val_loss: .5f}")
    logging.info(f"Val acc: {val_acc: .5f}, val f1: {val_f1: .5f}")
    logging.info(f"Val confusion matrix:\n {val_conf_matrix} \n")
    logging.info(f"Test loss: {test_loss: .5f}")
    logging.info(f"Test acc: {test_acc: .5f}, test f1: {test_f1: .5f}")
    logging.info(f"Test confusion matrix:\n {test_conf_matrix} \n")

    logging.info("-" * 40 + f"Epoch {epoch + 1}" + "-" * 40)

    # write tensorboard train log
    if tensorboard_logging:
        tb_writer.add_scalar("Validation/Val loss", val_loss, epoch)
        tb_writer.add_scalar("Validation/Val accuracy", val_acc, epoch)
        tb_writer.add_scalar("Validation/Val F1 score", val_f1, epoch)
        tb_writer.add_scalar("Evaluation/Test loss", test_loss, epoch)
        tb_writer.add_scalar("Evaluation/Test accuracy", test_acc, epoch)
        tb_writer.add_scalar("Evaluation/Test F1 score", test_f1, epoch)

    return val_acc, val_loss
