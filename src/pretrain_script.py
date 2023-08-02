import time
import numpy as np
import logging

from train import train
from test import test
from eval_knn import eval_knn
from eval_cluster import eval_cluster
from eval_mod_cluster import eval_mod_cluster
from eval_tsne import eval_tsne
from params.base_params import parse_base_args
from params.params_util import set_auto_params
from params.finetune_configs import *
from params.pretrain_configs import *
from output_utils.schedule_log_utils import check_execution_flag, reset_execution_flag, update_finetune_result


def train_loop():
    """The main pretraining script"""
    # Run dataset
    for dataset in datasets:
        # Encoders [Transformer, DeepSense]
        for model in models:
            for lambda_type in lambda_types:
                for lambda_weight in lambda_types[lambda_type]:
                    # evaluate the model
                    try:
                        try:
                            # set args
                            args = parse_base_args("train")
                            args.dataset = dataset
                            args.model = model
                            args.learn_framework = "CMCV2"
                            args.stage = "pretrain"
                            args.tag = f"{dataset}_{model}_{lambda_type}_{lambda_weight}"
                            args = set_auto_params(args, lambda_type, lambda_weight)
                            train(args)

                        except KeyboardInterrupt:
                            print("Excution interrupted by user, terminating ...")
                            return
                    except Exception as e:
                        print("Error: ", e)
                        continue


if __name__ == "__main__":
    start = time.time()
    # Step 1: test the finetuned models
    train_loop()

    # Step 2: calculate the mean result
    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .4f} seconds")
