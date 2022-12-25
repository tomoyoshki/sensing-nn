import torch
import logging
import numpy as np
from tqdm import tqdm

# utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def eval_given_model(args, classifier, miss_simulator, dataloader, loss_func):
    """ Evaluate the performance on the given dataloader.

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
    """
    # set both the classifier and the miss_simulator (detector + handler) to eval mode
    classifier.eval()
    miss_simulator.eval()

    # set the noise mode if needed

    # if args.miss_generator == "NoisyGenerator":
        # miss_simulator.miss_generator.set_noise_mode(args.noise_mode)

    # iterate over all batches
    num_batches = len(dataloader)
    classifier_loss_list = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(dataloader), total=num_batches):
            labels = labels.to(args.device)
            args.labels = labels

            logits = classifier(data, miss_simulator)
            # if args.miss_handler == "AdAutoencoder":
            #     g_loss, d_loss = handler_loss
            #     handler_loss = g_loss
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

    return mean_classifier_loss, -1, mean_f1, mean_acc, conf_matrix


def val_and_logging(
    args,
    epoch,
    tb_writer,
    classifier,
    miss_simulator,
    val_dataloader,
    test_dataloader,
    classifier_loss_func,
    train_loss,
    tensorboard_logging=True,
    eval_detector=False,
):
    """ Validation and logging function.

    Args:
        tb_writer (_type_): _description_
        classifier (_type_): _description_
        miss_simulator (_type_): _description_
        val_dataloader (_type_): _description_
        classifier_dataloader (_type_): _description_
        classifier_loss_func (_type_): _description_
    """
    logging.info(f"Train classifier loss: {train_loss[0]: .5f}, train handler loss: {train_loss[1]: .5f} \n")
    if eval_detector:
        eval_miss_detector(args, "Train", miss_simulator)

    val_classifier_loss, val_handler_loss, val_f1, val_acc, val_conf_matrix = eval_given_model(
        args,
        classifier,
        miss_simulator,
        val_dataloader,
        classifier_loss_func,
    )
    logging.info(f"Val classifier loss: {val_classifier_loss: .5f}, val handler loss: {val_handler_loss: .5f}")
    logging.info(f"Val acc: {val_acc: .5f}, val f1: {val_f1: .5f}")
    logging.info(f"Val confusion matrix:\n {val_conf_matrix} \n")
    if eval_detector:
        eval_miss_detector(args, "Val", miss_simulator)

    test_classifier_loss, test_handler_loss, test_f1, test_acc, test_conf_matrix = eval_given_model(
        args, classifier, miss_simulator, test_dataloader, classifier_loss_func
    )
    logging.info(f"Test classifier loss: {test_classifier_loss: .5f}, test handler loss: {test_handler_loss: .5f}")
    logging.info(f"Test acc: {test_acc: .5f}, test f1: {test_f1: .5f}")
    logging.info(f"Test confusion matrix:\n {test_conf_matrix} \n")
    if eval_detector:
        eval_miss_detector(args, "Test", miss_simulator)

    logging.info("-" * 40 + f"Epoch {epoch + 1}" + "-" * 40)

    # # print resilient handler parameters
    # if "ResilientHandler" in args.miss_handler:
    #     print("----------------------Resilient Handler Parameters----------------------")
    #     for name, param in miss_simulator.miss_handler.named_parameters():
    #         print(f"{name} \n", param.data.cpu().numpy())
    #     print("------------------------------------------------------------------------")

    # write tensorboard train log
    if tensorboard_logging:
        tb_writer.add_scalar("Validation/Val classifier loss", val_classifier_loss, epoch)
        tb_writer.add_scalar("Validation/Val handler loss", val_handler_loss, epoch)
        tb_writer.add_scalar("Validation/Val accuracy", val_acc, epoch)
        tb_writer.add_scalar("Validation/Val F1 score", val_f1, epoch)
        tb_writer.add_scalar("Evaluation/Test classifier loss", test_classifier_loss, epoch)
        tb_writer.add_scalar("Evaluation/Test handler loss", test_handler_loss, epoch)
        tb_writer.add_scalar("Evaluation/Test accuracy", test_acc, epoch)
        tb_writer.add_scalar("Evaluation/Test F1 score", test_f1, epoch)

    if eval_detector:
        return
    else:
        return val_acc


def eval_miss_detector(args, option, miss_simulator, eval_roc=False):
    """Evaluate the miss detector performance.

    Args:
        args (_type_): _description_
        classifier (_type_): _description_
        miss_simulator (_type_): _description_
        dataloader (_type_): _description_
    """
    # set both the classifier and the miss_simulator (detector + handler) to eval mode
    miss_simulator.eval()
    det_acc, det_f1, det_conf_matrix, det_auc, mean_loss = miss_simulator.eval_miss_detector(
        eval_roc,
        eval_auc_score=(option != "Train"),
        save_emb=args.save_emb,
    )

    if args.option == "train":
        logging.info(f"{option} miss detection AUC score: {det_auc: .5f}")
    else:
        print((f"{option} miss detection AUC score: {det_auc: .5f}, mean loss: {mean_loss: .5f}"))

    return det_auc, mean_loss
