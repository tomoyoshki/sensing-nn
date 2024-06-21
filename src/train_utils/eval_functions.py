import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn

# utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from train_utils.knn import extract_sample_features
from train_utils.loss_calc_utils import calc_pretrain_loss   

def eval_task_metrics(args, labels, predictions, regression=False):
    """Evaluate the downstream task metrics."""
    # different acc and f1-score definitions for single-class and multi-class problems
    if regression:
        mae = mean_absolute_error(labels, predictions)
        return [mae]
    else:
        acc_scores = []
        f1_scores = []

        if args.multi_class and False:
            for i in range(args.num_class):
                acc = accuracy_score(labels[:, i], predictions[:, i])
                f1 = f1_score(labels[:, i], predictions[:, i], zero_division=1)

                if acc <= 1 and f1 <= 1:
                    acc_scores.append(acc)
                    f1_scores.append(f1)

                mean_acc = np.mean(acc_scores)
                mean_f1 = np.mean(f1_scores)
                conf_matrix = []
        else:
            if args.task in {"distance_classification", "speed_classification"}:
                num_classes = args.dataset_config[args.task]["num_classes"]
                mean_acc = 1 - (np.abs(labels-predictions) / np.maximum(labels, (num_classes - 1) - labels))
                mean_acc = np.nan_to_num(mean_acc, nan=1.0)
                mean_acc = mean_acc.mean()
            else:    
                mean_acc = accuracy_score(labels, predictions)
            mean_f1 = f1_score(labels, predictions, average="macro", zero_division=1)
            try:
                conf_matrix = confusion_matrix(labels, predictions)
            except:
                conf_matrix = []

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

    all_detections_predictions = []
    all_detections_labels = []

    total = 0
    reduced = 0
    with torch.no_grad():
        for i, (time_loc_inputs, labels, detection_labels, index) in tqdm(enumerate(dataloader), total=num_batches):

            total += labels.shape[0]

            if "dual" in args.finetune_tag:
                labels = torch.stack((labels, detection_labels), dim=1) # [128, 2]

            # move to target device, FFT, and augmentations
            freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(freq_loc_inputs)

            if args.multi_class or "multiclass" in args.finetune_tag:
                labels = labels.reshape(labels.shape[0], -1)
                detection_labels = labels[:, -1].reshape(labels.shape[0], -1).long()
                labels = torch.nn.functional.one_hot(labels[:, 0], num_classes=args.num_class).float()
                if "dual" in args.finetune_tag:
                    labels = torch.cat((labels, detection_labels), dim=1)

            classifier_loss_list.append(loss_func(logits, labels).item())

            if "regression" in args.task:
                predictions = logits.squeeze()
            elif "dual" in args.finetune_tag:
                # First make prediction on if the car is present < 15 vs > 70
                detection_labels = labels[:, -1]
                detection_logits = logits[:, -2:] # last two class
                class_labels = labels[:, :-1] # class labels
                class_logits = logits[:, :-2]

                detection_prediction = detection_logits.argmax(dim=1, keepdim=False)
                detection_labels = detection_labels


                all_detections_labels.append(detection_labels.cpu().numpy())
                all_detections_predictions.append(detection_prediction.cpu().numpy())

                prediction_mask = (detection_prediction > 0)

                reduced += prediction_mask.shape[0] - prediction_mask.sum()
                class_logits = class_logits[prediction_mask]
                class_labels = class_labels[prediction_mask]

                max_logits = class_logits.max(dim=1)[0]
                # prediction_mask = (max_logits > 0.7)
                # reduced += prediction_mask.shape[0] - prediction_mask.sum()
                # class_logits = class_logits[prediction_mask]
                # class_labels = class_labels[prediction_mask]

                # predictions = (logits > 0.5).float()
                predictions = class_logits.argmax(dim=1, keepdim=False)
                labels = class_labels.argmax(dim=1, keepdim=False)
            else:
                if args.multi_class:
                    predictions = (logits > 0.85).float()
                    predictions = logits.argmax(dim=1, keepdim=False)
                    labels = labels.argmax(dim=1, keepdim=False) if labels.dim() > 1 else labels
                else:
                    # logits[logits < 0.7] = 0
                    # logits_sum = logits.sum(dim=1)
                    # logits_mask = (logits_sum == 0).int() # only keep the highest predictions
                    # logits[logits_mask, -1] = 1.0
                    predictions = logits.argmax(dim=1, keepdim=False) # predict whether each sample has car or not
                    # predictions[predictions > 3] = 3
                    labels = labels.argmax(dim=1, keepdim=False) if labels.dim() > 1 else labels

            # for future computation of acc or F1 score
            saved_predictions = predictions.cpu().numpy()
            saved_labels = labels.cpu().numpy()
            all_predictions.append(saved_predictions)
            all_labels.append(saved_labels)

    print(f"Total: {total} - Reduced: {reduced}")
    # calculate mean loss
    mean_classifier_loss = np.mean(classifier_loss_list)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    all_detections_predictions = np.concatenate(all_detections_predictions) if len(all_detections_predictions) > 0 else np.array([])
    all_detections_labels = np.concatenate(all_detections_labels) if len(all_detections_labels) > 0 else np.array([])

    # calculate the classification metrics
    metrics = eval_task_metrics(args, all_labels, all_predictions, regression=("regression" in args.task))

    if f"dual" in args.finetune_tag:
        detec_acc = accuracy_score(all_detections_labels, all_detections_predictions)
        detec_f1 = f1_score(all_detections_labels, all_detections_predictions, average="macro", zero_division=1)
        conf_matrix = confusion_matrix(all_detections_labels, all_detections_predictions)

        logging.info(f"Detection Accuracy: {detec_acc}, Detection F1: {detec_f1}")
        logging.info(f"Detection Confusion Matrix:\n{conf_matrix}")

        print(f"Detection Accuracy: {detec_acc}, Detection F1: {detec_f1}")
        print(f"Detection Confusion Matrix:\n{conf_matrix}")

    return mean_classifier_loss, metrics


def eval_pretrained_model(args, default_model, estimator, augmenter, dataloader, loss_func):
    """Evaluate the downstream task performance with KNN estimator."""
    default_model.eval()

    sample_embeddings = []
    labels = []
    loss_list = []
    with torch.no_grad():
        for time_loc_inputs, label, index in tqdm(dataloader, total=len(dataloader)):
            """Move idx to target device, save label"""
            index = index.to(args.device)
            label = label.argmax(dim=1, keepdim=False) if label.dim() > 1 and label.shape[1] > 1 else label.reshape(-1)
            labels.append(label.cpu().numpy())

            """Eval pretrain loss."""
            loss = calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs, index).item()
            loss_list.append(loss)

            """Eval KNN estimator."""
            aug_freq_loc_inputs = augmenter.forward("no", time_loc_inputs)
            feat = extract_sample_features(args, default_model.backbone, aug_freq_loc_inputs)
            sample_embeddings.append(feat.detach().cpu().numpy())

    # knn predictions
    sample_embeddings = np.concatenate(sample_embeddings)
    labels = np.concatenate(labels)
    predictions = torch.Tensor(estimator.predict(sample_embeddings))
    predictions = predictions.argmax(dim=1, keepdim=False) if predictions.dim() > 1 else predictions

    # compute metrics
    mean_loss = np.mean(loss_list)
    metrics = eval_task_metrics(args, labels, predictions, regression=("regression" in args.task))

    return mean_loss, metrics


def eval_predictive_task(args, default_model, augmenter, dataloader):
    """
    Evaluate the proxy predictive task performance during pretraining.
    """
    default_model.eval()

    # iterate over all batches
    num_batches = len(dataloader)
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for i, (time_loc_inputs, _, _) in tqdm(enumerate(dataloader), total=num_batches):
            # random augmentation with corresponding labels
            aug_freq_loc_inputs, pretrain_labels = augmenter.forward(
                "random",
                time_loc_inputs,
                return_aug_id=True if args.learn_framework == "MTSS" else False,
                return_aug_mods=True if args.learn_framework in {"ModPred", "ModPredFusion"} else False,
            )

            # forward pass
            pretrain_logits = default_model(aug_freq_loc_inputs)

            # get the predictions from the logits
            pretrain_predictions = (
                (nn.Sigmoid()(pretrain_logits) > 0.5).float()
                if args.learn_framework in {"ModPred", "ModPredFusion"}
                else pretrain_logits.argmax(dim=1, keepdim=False)
            )

            # for future computation of acc or F1 score
            saved_predictions = pretrain_predictions.cpu().numpy()
            saved_labels = pretrain_labels.cpu().numpy()
            all_predictions.append(saved_predictions)
            all_labels.append(saved_labels)

    # calculate mean loss
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # calculate the classification metrics
    mean_acc, mean_f1, conf_matrix = eval_task_metrics(args, all_labels, all_predictions)

    return mean_acc, mean_f1, conf_matrix


def eval_generative_model(args, model, estimator, augmenter, data_loader, loss_func):
    """Evaluate the performance with KNN estimator."""
    model.eval()

    loss_list = []
    with torch.no_grad():
        for time_loc_inputs, label, index in tqdm(data_loader, total=len(data_loader)):
            """Eval contrastive loss."""
            aug_freq_loc_inputs_1 = augmenter.forward("random", time_loc_inputs)
            decoded_x, padded_x, masks = model(aug_freq_loc_inputs_1)

            # forward pass
            loss = loss_func(padded_x, decoded_x, masks).item()
            loss_list.append(loss)

    mean_loss = np.mean(loss_list)

    return mean_loss, -1, -1, None


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
    if args.train_mode in {"contrastive", "predictive", "generative"} and args.stage == "pretrain":
        logging.info(f"Train {args.train_mode} loss: {train_loss: .5f} \n")
    else:
        logging.info(f"Training loss: {train_loss: .5f} \n")

    if args.train_mode == "supervised" or args.stage == "finetune":
        """Supervised training or fine-tuning"""
        val_loss, val_metrics = eval_supervised_model(args, model, augmenter, val_loader, loss_func)
        test_loss, test_metrics = eval_supervised_model(args, model, augmenter, test_loader, loss_func)
    else:
        """Predictive pretrain task"""
        if args.train_mode == "predictive":
            val_pretrain_acc, val_pretrain_f1, val_pretrain_conf_matrix = eval_predictive_task(
                args, model, augmenter, val_loader
            )
            logging.info(f"Val pretrain acc: {val_pretrain_acc: .5f}, val pretrain f1: {val_pretrain_f1: .5f}")
            logging.info(f"Val pretrain confusion matrix:\n {val_pretrain_conf_matrix} \n")
            test_pretrain_acc, test_pretrain_f1, test_pretrain_conf_matrix = eval_predictive_task(
                args, model, augmenter, test_loader
            )
            logging.info(f"Test pretrain acc: {test_pretrain_acc: .5f}, test pretrain f1: {test_pretrain_f1: .5f}")
            logging.info(f"Test pretrain confusion matrix:\n {test_pretrain_conf_matrix} \n")

            if tensorboard_logging:
                tb_writer.add_scalar("Evaluation/Pretrain Test accuracy", test_pretrain_acc, epoch)
                tb_writer.add_scalar("Evaluation/Pretrain Test F1 score", test_pretrain_f1, epoch)

        """All self-supervised pre-training tasks"""
        val_loss, val_metrics = eval_pretrained_model(args, model, estimator, augmenter, val_loader, loss_func)
        test_loss, test_metrics = eval_pretrained_model(args, model, estimator, augmenter, test_loader, loss_func)

    if "regression" in args.task:
        logging.info(f"Val loss: {val_loss: .5f}, val mae: {val_metrics[0]: .5f}")
        logging.info(f"Test loss: {test_loss: .5f}, test mae: {test_metrics[0]: .5f}")
    else:
        logging.info(f"Val loss: {val_loss: .5f}")
        logging.info(f"Val acc: {val_metrics[0]: .5f}, val f1: {val_metrics[1]: .5f}")
        logging.info(f"Val confusion matrix:\n {val_metrics[2]} \n")
        logging.info(f"Test loss: {test_loss: .5f}")
        logging.info(f"Test acc: {test_metrics[0]: .5f}, test f1: {test_metrics[1]: .5f}")
        logging.info(f"Test confusion matrix:\n {test_metrics[2]} \n")

    # write tensorboard train log
    if tensorboard_logging:
        tb_writer.add_scalar("Validation/Val loss", val_loss, epoch)
        tb_writer.add_scalar("Evaluation/Test loss", test_loss, epoch)
        if "regression" in args.task:
            tb_writer.add_scalar("Validation/Val mae", val_metrics[0], epoch)
            tb_writer.add_scalar("Evaluation/Test mae", test_metrics[0], epoch)
        else:
            tb_writer.add_scalar("Validation/Val accuracy", val_metrics[0], epoch)
            tb_writer.add_scalar("Validation/Val F1 score", val_metrics[1], epoch)
            tb_writer.add_scalar("Evaluation/Test accuracy", test_metrics[0], epoch)
            tb_writer.add_scalar("Evaluation/Test F1 score", test_metrics[1], epoch)

    return val_metrics[0], val_loss
