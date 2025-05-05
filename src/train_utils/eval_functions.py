import torch
import logging
import numpy as np
from tqdm import tqdm

# utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error

def eval_task_metrics(args, labels, predictions, regression=False):
    """Evaluate the downstream task metrics."""
    # different acc and f1-score definitions for single-class and multi-class problems
    if regression:
        mae = mean_absolute_error(labels, predictions)
        return [mae]
    else:
        acc_scores = []
        f1_scores = []

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
            except ValueError:
                conf_matrix = []
        # breakpoint()
        return mean_acc, mean_f1, conf_matrix


def eval_supervised_model_float(args, classifier, augmenter, dataloader, loss_func):
    """Evaluate the performance on the given dataloader.

    Args:
        args: Runtime arguments
        classifier: The classifier model
        augmenter: Data augmentation module
        dataloader: Data loader for evaluation
        loss_func: Loss function
    """
    # set classifier to eval mode
    classifier.eval()

    # iterate over all batches
    classifier_loss_list = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, (time_loc_inputs, labels, index) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # move to target device, FFT, and augmentations
            freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(freq_loc_inputs)
            classifier_loss_list.append(loss_func(logits, labels).item())

            if "regression" in args.task:
                predictions = logits.squeeze()
                # Take the maximum prediction across time dimension if needed
                if predictions.dim() > 1:
                    predictions = torch.max(predictions, dim=1)[0]
            else:
                if args.multi_class:
                    predictions = (logits > 0.5).float()
                    # Take the maximum prediction across time dimension if needed
                    if predictions.dim() > 2:
                        predictions = torch.max(predictions, dim=1)[0]
                else:
                    predictions = logits.argmax(dim=1)
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=1)

            # for future computation of metrics
            saved_predictions = predictions.cpu().numpy()
            saved_labels = labels.cpu().numpy()
            all_predictions.append(saved_predictions)
            all_labels.append(saved_labels)

    # calculate mean loss and combine predictions
    mean_classifier_loss = np.mean(classifier_loss_list)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # calculate the metrics
    metrics = eval_task_metrics(args, all_labels, all_predictions, regression=("regression" in args.task))
    
    # Create metrics dictionary similar to quantized version
    if "regression" in args.task:
        metrics_dict = {"mean_mae": metrics[0]}
    else:
        metrics_dict = {
            "mean_acc": metrics[0],
            "mean_f1": metrics[1],
            "confusion_matrix": metrics[2]
        }
    # Log metrics
    logging.info(f"\nMetrics for full precision model:")
    logging.info(f"Loss: {mean_classifier_loss:.4f}")
    for metric_name, metric_value in metrics_dict.items():
        if metric_name == "confusion_matrix":
            # Convert to integers and disable scientific notation
            np.set_printoptions(suppress=True)
            confusion_matrix_int = metric_value.astype(int)
            logging.info(f"{metric_name}:\n{confusion_matrix_int}")
        else:
            logging.info(f"{metric_name}: {metric_value:.4f}")

    # return metrics_per_bitwidth
    return metrics_dict[0], mean_classifier_loss # Accuracy, Loss

def get_random_quantizations_schemes_given_average(args, classifier):
    quantization_config = args.dataset_config["quantization"]
    average_bitwidths = quantization_config["test_and_val_average_bitwidths"]
    bitwidth_options = quantization_config["bitwidth_options"]
    number_of_quantization_schemes_per_avg_bw = quantization_config["test_and_val_total_schemes"]
    delta = quantization_config["delta"]
    num_layers = len(classifier.Conv.layer_registry)
    
    # Initialize dictionary to store quantization schemes for each target average bitwidth
    quantization_schemes = {avg_bw: [] for avg_bw in average_bitwidths}
    
    # For each target average bitwidth
    logging.info("-"*30 + "Started Generating Random Bitwidth Configurations" + "-"*30)
    for target_avg_bw in average_bitwidths:
       
        # print("Target Average Bitwidth: ", target_avg_bw)
        # print(f"Number of Quantization Schemes: {number_of_quantization_schemes_per_avg_bw}")
        while len(quantization_schemes[target_avg_bw]) < number_of_quantization_schemes_per_avg_bw:
            # Randomly generate bitwidths for each layer
            # print(f"Number of Schemes Generated: {len(quantization_schemes[target_avg_bw])}", end="\n")
            # breakpoint()
            # layer_bitwidths = np.random.choice(bitwidth_options, size=num_layers)
            # current_avg_bw = np.mean(layer_bitwidths)
            # print(current_avg_bw, end=", ")
            # breakpoint()

            # Method 1: Start with a uniform configuration close to target
            if np.random.random() < 0.5:
                # Find the closest available bitwidth options to the target
                closest_lower = max([b for b in bitwidth_options if b <= target_avg_bw], default=min(bitwidth_options))
                closest_higher = min([b for b in bitwidth_options if b >= target_avg_bw], default=max(bitwidth_options))
                
                # Determine the ratio of higher to lower bitwidths needed
                if closest_lower == closest_higher:
                    layer_bitwidths = np.full(num_layers, closest_lower)
                else:
                    num_higher = int(round(num_layers * (target_avg_bw - closest_lower) / (closest_higher - closest_lower)))
                    layer_bitwidths = np.array([closest_higher] * num_higher + [closest_lower] * (num_layers - num_higher))
                    np.random.shuffle(layer_bitwidths)  # Shuffle to randomize which layers get which bitwidth
            
            # Method 2: Sample around the target with a normal distribution
            else:
                # Find standard deviation based on available bitwidth range
                std_dev = max(1.0, (max(bitwidth_options) - min(bitwidth_options)) / 4)
                
                # Generate normally distributed values around the target
                raw_values = np.random.normal(target_avg_bw, std_dev, num_layers)
                
                # Map each value to the nearest available bitwidth option
                layer_bitwidths = np.array([min(bitwidth_options, key=lambda x: abs(x - val)) for val in raw_values])
            
            current_avg_bw = np.mean(layer_bitwidths)
            # Chneck if the average is within the acceptable range
            if abs(current_avg_bw - target_avg_bw) <= delta:
                # Create dictionary mapping layer names to bitwidths
                scheme = {}
                for i, layer_name in enumerate(classifier.Conv.layer_registry.keys()):
                    scheme[layer_name] = int(layer_bitwidths[i])
                
                # Add scheme if it's not already in the list
                if scheme not in quantization_schemes[target_avg_bw]:
                    quantization_schemes[target_avg_bw].append(scheme)
        logging.info(f"Done: Target Average Bitwidth= {target_avg_bw}")
    return quantization_schemes





def eval_quantized_supervised_model(args, classifier, augmenter, dataloader, loss_func, loss_type = "Validation", return_metric_type = "mean"):

    """Evaluate the performance on the given dataloader.

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
        return_metric_type (_type_): Can be "mean" or "best
    """
    # set both the classifier and augmenter to eval mode
    quantization_schemes = get_random_quantizations_schemes_given_average(args, classifier)
    classifier.eval()

    # iterate over all batches
    num_batches = len(dataloader)
    # classifier_loss_list = []
    # all_predictions = []
    # all_labels = []
    metrics_per_bitwidth = {}
    with torch.no_grad():
        for avg_bitwidth, schemes in tqdm(quantization_schemes.items(), total=len(quantization_schemes)):
            scheme_metrics = []
            classifier_loss_list = []
            all_predictions = []
            all_labels = []

            for scheme in schemes:
                for layer_name, bitwidth in scheme.items():
                    layer = classifier.Conv.layer_registry[layer_name]
                    layer.set_bitwidth(bitwidth)


                classifier_loss_list = []
                # final_predictions_curr_scheme = []
                # final_labels_curr_scheme = []
                for i, (time_loc_inputs, labels, index) in enumerate(dataloader):
                    # Move to target device, FFT, and augmentations
                    freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)
       
                    # Forward pass
                    logits, _ = classifier(freq_loc_inputs)
                    classifier_loss_list.append(loss_func(logits, labels).item())
                    # breakpoint()
                    if "regression" in args.task:
                        predictions = logits.squeeze()
                        # Take the maximum prediction across time dimension if needed
                        if predictions.dim() > 1:
                            predictions = torch.max(predictions, dim=1)[0]
                    else:
                        if args.multi_class:
                            predictions = (logits > 0.5).float()
                            # Take the maximum prediction across time dimension if needed
                            if predictions.dim() > 2:
                                predictions = torch.max(predictions, dim=1)[0]
                        else:
                            predictions = logits.argmax(dim=1)
                            
                            if labels.dim() > 1:
                                labels = labels.argmax(dim=1)
                    saved_predictions = predictions.cpu().numpy()
                    saved_labels = labels.cpu().numpy()
                    all_predictions.append(saved_predictions)
                    all_labels.append(saved_labels)
                
            # Calculate mean loss
            mean_classifier_loss = np.mean(classifier_loss_list)
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            metrics = eval_task_metrics(args, all_labels, all_predictions, regression=("regression" in args.task))
            metrics_dict = {"mean_acc": metrics[0], "mean_f1": metrics[1], "confusion_matrix": metrics[2], "mean_loss": mean_classifier_loss}
            # Store metrics for current bitwidth
            metrics_per_bitwidth[avg_bitwidth] = metrics_dict

            # Log metrics for current bitwidth
            logging.info(f"\nMetrics for average bitwidth {avg_bitwidth}:")
            logging.info(f"Loss: {mean_classifier_loss:.4f}")
            for metric_name, metric_value in metrics_dict.items():
                if metric_name == "confusion_matrix":
                    # Convert to integers and disable scientific notation
                    np.set_printoptions(suppress=True)
                    confusion_matrix_int = metric_value.astype(int)
                    logging.info(f"{metric_name}:\n{confusion_matrix_int}")
                else:
                    logging.info(f"{metric_name}: {metric_value:.4f}")


    # Calculate and Log average metrics across all schemes for each bitwidth

    if "regression" in args.task:
        metrics = [np.mean([m['metrics'][0] for m in metrics_per_bitwidth.values()])]
        # test_metrics = [np.mean([m['metrics'][0] for m in metrics_per_bitwidth.values()])]
    else:
        # Average the confusion matrices properly
        val_confusion_matrices = [m["confusion_matrix"] for m in metrics_per_bitwidth.values()]
        # test_confusion_matrices = [m['metrics'][2] for m in test_metrics_dict.values()]
        
        val_avg_conf_matrix = np.mean(val_confusion_matrices, axis=0)
        # test_avg_conf_matrix = np.mean(test_confusion_matrices, axis=0)

        average_metrics = {
            "mean_acc": np.mean([m["mean_acc"] for m in metrics_per_bitwidth.values()]),  # accuracy
            "mean_f1": np.mean([m["mean_f1"] for m in metrics_per_bitwidth.values()]),  # f1
            "mean_loss": np.mean([m["mean_loss"] for m in metrics_per_bitwidth.values()]),  # loss
            "confusion_matrix": val_avg_conf_matrix
        }
        
        logging.info(f"\nAverage metrics across all schemes for each bitwidth:")
        for metric_name, metric_value in average_metrics.items():
            if metric_name == "confusion_matrix":
                # Convert to integers and disable scientific notation
                np.set_printoptions(suppress=True)
                confusion_matrix_int = metric_value.astype(int)
                logging.info(f"Average {metric_name}:\n{confusion_matrix_int}")
            else:
                logging.info(f"Average {metric_name}: {metric_value:.4f}")

        logging.info(f"Bitwidths with best metrics:")

        best_bitwidth = max(metrics_per_bitwidth.items(), 
                       key=lambda x: x[1]['mean_acc'])  # using mean_acc from metrics_dict
        logging.info(f"\nBest performing bitwidth: {best_bitwidth[0]} with accuracy: {best_bitwidth[1]['mean_acc']:.4f}")

        best_bitwidth_lowest_loss = min(metrics_per_bitwidth.items(),
                       key=lambda x: x[1]['mean_loss'])
        logging.info(f"\nBest performing bitwidth with lowest loss: {best_bitwidth_lowest_loss[0]} with loss: {best_bitwidth_lowest_loss[1]['mean_loss']:.4f}")

        if return_metric_type == "best":
            return best_bitwidth[1]['mean_acc'], best_bitwidth_lowest_loss[1]['mean_loss']
        elif return_metric_type == "mean":
            return average_metrics["mean_acc"], average_metrics["mean_loss"]


def eval_quantized_supervised_model_self_selection(args, classifier, augmenter, dataloader, loss_func, loss_type = "Validation"):
    """Evaluate the performance on the given dataloader.

    Args:
        args: Runtime arguments
        classifier: The classifier model
        augmenter: Data augmentation module
        dataloader: Data loader for evaluation
        loss_func: Loss function
    """
    # set classifier to eval mode
    classifier.eval()

    # iterate over all batches
    classifier_loss_list = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, (time_loc_inputs, labels, index) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # move to target device, FFT, and augmentations
            freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            try:
                logits,_ = classifier(freq_loc_inputs)
                # breakpoint()
                classifier_loss_list.append(loss_func(logits, labels).item())
            except Exception as e:
                logging.error(f"Error during forward pass: {e}")
                breakpoint()
                raise
            if "regression" in args.task:
                predictions = logits.squeeze()
                # Take the maximum prediction across time dimension if needed
                if predictions.dim() > 1:
                    predictions = torch.max(predictions, dim=1)[0]
            else:
                if args.multi_class:
                    predictions = (logits > 0.5).float()
                    # Take the maximum prediction across time dimension if needed
                    if predictions.dim() > 2:
                        predictions = torch.max(predictions, dim=1)[0]
                else:
                    predictions = logits.argmax(dim=1)
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=1)

            # for future computation of metrics
            saved_predictions = predictions.cpu().numpy()
            saved_labels = labels.cpu().numpy()
            all_predictions.append(saved_predictions)
            all_labels.append(saved_labels)

    # calculate mean loss and combine predictions
    mean_classifier_loss = np.mean(classifier_loss_list)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # calculate the metrics
    metrics = eval_task_metrics(args, all_labels, all_predictions, regression=("regression" in args.task))
    
    # Create metrics dictionary similar to quantized version
    if "regression" in args.task:
        metrics_dict = {"mean_mae": metrics[0]}
    else:
        metrics_dict = {
            "mean_acc": metrics[0],
            "mean_f1": metrics[1],
            "confusion_matrix": metrics[2],
            "mean_loss": mean_classifier_loss
        }
    # Log metrics
    logging.info(f"\nMetrics for model with self-selection bitwidth using Gumbal Softmax Sampling:")
    logging.info(f"{loss_type} Loss: {mean_classifier_loss:.4f}")
    for metric_name, metric_value in metrics_dict.items():
        if metric_name == "confusion_matrix":
            # Convert to integers and disable scientific notation
            np.set_printoptions(suppress=True)
            confusion_matrix_int = metric_value.astype(int)
            logging.info(f"{metric_name}:\n{confusion_matrix_int}")
        else:
            logging.info(f"{metric_name}: {metric_value:.4f}")

    # return metrics_per_bitwidth
    return metrics_dict["mean_acc"], mean_classifier_loss # Accuracy, Loss


def eval_quantized_supervised_model_self_selection_per_test(args, classifier, augmenter, dataloader, loss_func, loss_type = "Validation"):
    """Evaluate the performance on the given dataloader.

    Args:
        args: Runtime arguments
        classifier: The classifier model
        augmenter: Data augmentation module
        dataloader: Data loader for evaluation
        loss_func: Loss function
    """
    # set classifier to eval mode
    classifier.eval()

    # iterate over all batches
    classifier_loss_list = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        conv_classifier = classifier.Conv
        conv_classifier.set_fixed_tiled_bitwidths()
        for i, (time_loc_inputs, labels, index) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # move to target device, FFT, and augmentations
            freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            try:
                logits,_ = classifier(freq_loc_inputs)
                # breakpoint()
                classifier_loss_list.append(loss_func(logits, labels).item())
            except Exception as e:
                logging.error(f"Error during forward pass: {e}")
                breakpoint()
                raise
            if "regression" in args.task:
                predictions = logits.squeeze()
                # Take the maximum prediction across time dimension if needed
                if predictions.dim() > 1:
                    predictions = torch.max(predictions, dim=1)[0]
            else:
                if args.multi_class:
                    predictions = (logits > 0.5).float()
                    # Take the maximum prediction across time dimension if needed
                    if predictions.dim() > 2:
                        predictions = torch.max(predictions, dim=1)[0]
                else:
                    predictions = logits.argmax(dim=1)
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=1)

            # for future computation of metrics
            saved_predictions = predictions.cpu().numpy()
            saved_labels = labels.cpu().numpy()
            all_predictions.append(saved_predictions)
            all_labels.append(saved_labels)

    # calculate mean loss and combine predictions
    mean_classifier_loss = np.mean(classifier_loss_list)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # calculate the metrics
    metrics = eval_task_metrics(args, all_labels, all_predictions, regression=("regression" in args.task))
    
    # Create metrics dictionary similar to quantized version
    if "regression" in args.task:
        metrics_dict = {"mean_mae": metrics[0]}
    else:
        metrics_dict = {
            "mean_acc": metrics[0],
            "mean_f1": metrics[1],
            "confusion_matrix": metrics[2],
            "mean_loss": mean_classifier_loss
        }
    # Log metrics
    logging.info(f"\nMetrics for model with self-selection bitwidth using Gumbal Softmax Sampling:")
    logging.info(f"{loss_type} Loss: {mean_classifier_loss:.4f}")
    for metric_name, metric_value in metrics_dict.items():
        if metric_name == "confusion_matrix":
            # Convert to integers and disable scientific notation
            np.set_printoptions(suppress=True)
            confusion_matrix_int = metric_value.astype(int)
            logging.info(f"{metric_name}:\n{confusion_matrix_int}")
        else:
            logging.info(f"{metric_name}: {metric_value:.4f}")

    # return metrics_per_bitwidth
    return metrics_dict["mean_acc"], mean_classifier_loss # Accuracy, Loss

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
    """Supervised training or fine-tuning"""
    quantization_config = args.dataset_config["quantization"]

    if not quantization_config["enable"]:
        logging.info("Evaluating full precision model - Float32")
        val_acc, val_loss = eval_supervised_model_float(args, model, augmenter, val_loader, loss_func)
        test_loss, test_metrics = eval_supervised_model_float(args, model, augmenter, test_loader, loss_func)
    elif quantization_config["enable"]:
        logging.info("Evaluating quantized model")
        # Check if the model is quantized
        if quantization_config["testing_type"] == "mean_mixed_precision":
            # Use the quantized model for evaluation
            val_acc, val_loss = eval_quantized_supervised_model(args, model, augmenter, val_loader, loss_func)
            test_loss, test_metrics = eval_quantized_supervised_model(args, model, augmenter, test_loader, loss_func)
        elif quantization_config["testing_type"] == "model_self_selection_per_forward_pass":
            logging.info("Evaluating model with self-selection bitwidth using Gumbal Softmax Sampling")
            val_acc, val_loss = eval_quantized_supervised_model_self_selection(args, model, augmenter, val_loader, loss_func)
            test_loss, test_metrics = eval_quantized_supervised_model_self_selection(args, model, augmenter, test_loader, loss_func, loss_type="Test")
        elif quantization_config["testing_type"] == "model_self_selection_per_test":
            logging.info("Evaluating model with self-selection bitwidth only sampling once per test using Gumbal Softmax Sampling")
            val_acc, val_loss = eval_quantized_supervised_model_self_selection_per_test(args, model, augmenter, val_loader, loss_func)
            test_loss, test_metrics = eval_quantized_supervised_model_self_selection_per_test(args, model, augmenter, test_loader, loss_func, loss_type="Test")
        else:
            raise ValueError(f"Error in val_and_logging: No valid condition given for evaluation")
    else:
        raise ValueError(f"Error in val_and_logging: No valid condition given for evaluation")
    
    #  # write tensorboard train log
    #     if tensorboard_logging:
    #         tb_writer.add_scalar("Validation/Val loss", val_loss, epoch)
    #         tb_writer.add_scalar("Evaluation/Test loss", test_loss, epoch)
    #         if "regression" in args.task:
    #             tb_writer.add_scalar("Validation/Val mae", val_metrics[0], epoch)
    #             tb_writer.add_scalar("Evaluation/Test mae", test_metrics[0], epoch)
    #         else:
    #             tb_writer.add_scalar("Validation/Val accuracy", val_metrics[0], epoch)
    #             tb_writer.add_scalar("Validation/Val F1 score", val_metrics[1], epoch)
    #             tb_writer.add_scalar("Evaluation/Test accuracy", test_metrics[0], epoch)
    #             tb_writer.add_scalar("Evaluation/Test F1 score", test_metrics[1], epoch)

    return val_acc, val_loss