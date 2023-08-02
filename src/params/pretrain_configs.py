"""
Store the finetune configs here for the consistency between test and finetune.
"""

datasets = [
    # "ACIDS",
    # "Parkland",
    # "RealWorld_HAR",
    "PAMAP2",
]

models = [
    "TransformerV4",
    # "DeepSense",
]

lambda_types = {
    # "private_contrastive_loss_weight": [
    #     0.1, 
    #     3,
    #     10
    # ],
    # "orthogonal_loss_weight": [
    #     0.1,
    #     1,
    #     10
    # ],
    # "rank_loss_weight": [
    #     0.1,
    #     1,
    #     3,
    #     10
    # ]
}

margin_weights = [
    0.05,
    0.5,
    5
]


