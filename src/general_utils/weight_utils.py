import os
import torch


def load_model_weight(model, weight_file):
    """Load the trained model weight into the model.

    Args:
        model (_type_): _description_
        weight_file (_type_): _description_
    """
    trained_dict = torch.load(weight_file)
    model_dict = model.state_dict()
    load_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)

    return model
