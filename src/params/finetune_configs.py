"""
Store the finetune configs here for the consistency between test and finetune.
"""

datasets = [
    "ACIDS",
    "Parkland",
    "RealWorld_HAR",
    "PAMAP2",
]

models = [
    "TransformerV4",
    "DeepSense",
]

learn_frameworks = [
    "SimCLR",
    "MoCo",
    "CMC",
    "CMCV2",
    "MAE",
    "Cosmo",
    "Cocoa",
    "MTSS",
    "TS2Vec",
    "GMC",
    "TNC",
    "TSTCC",
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

label_ratios = [
    1.0,
    0.1,
    0.01,
]

runs = 5
