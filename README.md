# IoT Sensing Training Pipelines

## Requirements

1. **Dependencies**: ```conda create -n [NAME] python=3.10; conda activate [NAME]```

2. **Installation**:

    ```bash
    git clone [repo] [dir]
    cd [dir]
    pip install -r requirements.txt
    ```

## Datasets

There are four datasets available. Parkland (MOD, parkland is how we call it, but in paper, please refer to it as MOD), ACIDS, RealWorld-HAR, PAMAP2.

Specify which datasets to run.

```bash
python3 train.py -dataset=[Parkland, ACIDS, RealWorld_HAR, PAMAP2]
```


## Models

Models and relevant utils functions are implemented in `src/models/`. If you are adding a new model, please read the followings. I have also added exception message to inform you if you miss any of these. 

**Model configurations**

1. Add configurations in your dataset config file (`src/data/DATASET.yaml`), this will be the configurations of your models. See how DeepSense or TransformerV4 in each dataset.yaml file. 

**Model Registry**

1. Implement your model
2. Import your model and add model in `src/train_utils/model_selection.py`

I have created a NEWMODEL as an example with PAMAP2.yaml config file. 

### Experiment Weights

The code will automatically select the newest weights, or specifiy the weight path (see below)
```
└── weights
    └── DATASET_MODEL_debug
        ├── expXXX_supervised_TASK_1.0
        └── exp[LATEST]_supervised_TASK_1.0
```

## Running the code


### Supervised training

```bash
python3 train.py -gpu=X -model=X -dataset=X
```

**Example**

```bash
python3 train.py -gpu=0 -model=TransformerV4 -dataset=Parkland
```


### Testing


```bash
python3 test.py -gpu=X -model=X -dataset=X
```

**Example**: This will select the latest experiment. 

```bash
python3 train.py -gpu=0 -model=TransformerV4 -dataset=Parkland
```

**Example**: You can also specify a specific experiment to run. . 

```bash
python3 train.py -gpu=0 -model=TransformerV4 -dataset=Parkland -weight=/home/tkimura4/emsoft/weights/Parkland_TransformerV4_debug/exp9_supervised_vehicle_classification_1.0
```
