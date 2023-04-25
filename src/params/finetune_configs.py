"""
Store the finetune configs here for the consistency between test and finetune.
"""

datasets = [
<<<<<<< HEAD
    "ACIDS",
    # "Parkland",
=======
    # "ACIDS",
    "Parkland",
>>>>>>> 99bb66c3172973fd94c290f5e69e0c6bb47d1aee
    # "RealWorld_HAR",
    # "PAMAP2",
]

models = [
    # "TransformerV4",
    "DeepSense",
]

learn_frameworks = [
    # "SimCLR",
    # "MoCo",
    # "CMC",
    # "CMCV2",
    # "MAE",
    # "Cosmo",
    # "Cocoa",
    # "MTSS",
    # "TS2Vec",
    "GMC",
    # "TNC",
    # "TSTCC",
]

tasks = {
    "ACIDS": [
        "vehicle_classification",
    ],
    "Parkland": [
        "vehicle_classification",
    ],
    "RealWorld_HAR": [
        "activity_classification",
    ],
    "PAMAP2": [
        "activity_classification",
    ],
}

label_ratios = {
    "fintune": [
        1.0,
        0.1,
        0.01,
    ],
    "knn": [
        1.0,
        0.1,
        0.01,
    ],
    "cluster": [
        1.0,
    ],
}

runs = {
    "finetune": 5,
    "knn": 5,
    "cluster": 1
}
