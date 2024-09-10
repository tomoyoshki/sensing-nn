# FOCAL (VibroFM)

## Requirements

1. **Dependencies**: ```conda create -n [NAME] python=3.10; conda activate [NAME]```

2. **Installation**:

    ```bash
    git clone [repo] [dir]
    cd [dir]
    pip install -r requirements.txt
    ```

## Data

Processed Data path: `/home/tkimura4/data/datasets/MOD/GracesQuarters/2024-08-0X-GQ/individual_time_samples`

Index file path: `/home/tkimura4/data/indices/single/2024-08-0X-GQ`

You do not need to modify these if you are running on Eugene. The file path is already in the `Parkland.yaml` file. 

In `Parkland.yaml`, there is a `SET` for each `TASK`, where the index path is specified.

```yaml
vehicle_classification:
    ...
    gcq20240806:
            num_classes: 4
            train_index_file: /home/tkimura4/data/indices/single/2024-08-06-GQ/train_index.txt
            val_index_file: /home/tkimura4/data/indices/single/2024-08-06-GQ/val_index.txt
            test_index_file: /home/tkimura4/data/indices/single/2024-08-06-GQ/test_index.txt

    gcq20240807:
            num_classes: 4
            train_index_file: /home/tkimura4/data/indices/single/2024-08-07-GQ/train_index.txt
            val_index_file: /home/tkimura4/data/indices/single/2024-08-07-GQ/val_index.txt
            test_index_file: /home/tkimura4/data/indices/single/2024-08-07-GQ/test_index.txt
    ...
distance_regression:
    ...
distance_classification:
    ...
```

## Models

The model we are using is `TransformerV4.py`, it is a SWIN-Transformer. 

### Pretrained Weights

The code will automatically select the newest weights, or specifiy the weight path (see below)
```
└── weights
    └── Parkland_TransformerV4_debug
        ├── exp0_contrastive_CMCV2
        ├── exp..._contrastive_CMCV2
        └── exp[LATEST]_contrastive_CMCV2
```

Put the weight folder under Parkland_TransformerV4_debug

## Training

The following command will run supervised training for Transformer (`-model=TransformerV4`).

### FOCAL Pre-training

```bash
python train.py -model=[MODEL] -dataset=[DATASET] -learn_framework=FOCAL
```

### FOCAL Fine-tuning with specific downstream task

```bash
python train.py -model=TransformerV4 -dataset=Parkland -learn_framework=CMCV2 -task=[TASK] -stage=finetune -model_weight=[PATH TO MODEL WEIGHT] -finetune_set=[SET]
```

- `dataset` default to Parkland
- `TASK` is default to vehicle_classification
- `MODEL WEIGHT` is default to the weight folder with latest `expID`
- Specify `finetune set` you would like to run
    - For GQ data we collected in august, it would be `gcq20240806` and `gcq20240806`

**Example**

- Run Aug20240806 Data Vehicle Classification

    `python3 train.py -gpu=0 -model=TransformerV4 -learn_framework=CMCV2 -stage=finetune -finetune_set=gcq20240806 -balanced_sample=False`

- Run Aug20240806 Data Distance Regression

    `python3 train.py -gpu=0 -model=TransformerV4 -learn_framework=CMCV2 -stage=finetune -finetune_set=gcq20240806 -balanced_sample=False -task=distance_regression`

- Run Aug20240807 Data Distance Regression with specific weights

    `python3 train.py -gpu=0 -model=TransformerV4 -learn_framework=CMCV2 -stage=finetune -finetune_set=gcq20240806 -balanced_sample=False -task=distance_regression -model_weight=/home/tkimura4/FTSSense/weights/Parkland_TransformerV4_debug/exp99_contrastive_CMCV2`

- Finetune weights should be stored under the same model weight folder

### Testing

**Note that for testing, simply change `train.py` to `test.py` and ensure everything else is the same, and add -test_set=XXX**