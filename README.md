# FoundationSense

Transformer-based foundation models for (multi-modal) time-series sensing data


### Supervised Learning 

#### Training
```
CUDA_VISIBLE_DEVICES=0 python3 train.py -dataset=Parkland -train_mode=supervised -task=vehicle_classification -model=TransformerV4
```

#### Testing 
```
CUDA_VISIBLE_DEVICES=0 python3 test.py -dataset=Parkland -train_mode=supervised -task=vehicle_classification -model=TransformerV4 -model_weight=/home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp10_supervised
```

### Contrastive Learning


#### Pretraining
```
CUDA_VISIBLE_DEVICES=0 python3 train.py -dataset=Parkland -train_mode=contrastive -contrastive_framework=SimCLR -stage=pretrain -model=TransformerV4
```

#### Finetuning

```
CUDA_VISIBLE_DEVICES=0 python3 train.py -dataset=Parkland -train_mode=contrastive -contrastive_framework=SimCLR -stage=finetune -task=vehicle_classification -model=TransformerV4
```

#### Testing 
```
CUDA_VISIBLE_DEVICES=0 python3 test.py -dataset=Parkland -train_mode=contrastive -contrastive_framework=SimCLR -stage=finetune -task=vehicle_classification -model=TransformerV4 -model_weight=/home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp22_contrastive
```

### Parkland Results
#### 1. Augmenter performance with supervised train
|  Date       | Model | Augmenter   |  Accuracy  | User| Weight Checkpoint |
| :---:       |    :----:    |    :----:   |      :---: |     :---: |    :--- | 
| 20230116    | TransformerV4 | PhaseShift         | 88.68%   | sl29 | Parkland_TransformerV4/exp6_supervised |
| 20230116    | TransformerV4 | FreqMask           | 85.53%   | sl29 | Parkland_TransformerV4/exp9_supervised |
| 20230116    | TransformerV4 | TimeMask           | 72.94%   | sl29 |  Parkland_TransformerV4/exp7_supervised |
| 20230116    | TransformerV4 | HorizontalFlip     | 84.59%   | sl29 |  Parkland_TransformerV4/exp10_supervised|
| 20230116    | TransformerV4 | Jitter             | 84.46%   | sl29 |  Parkland_TransformerV4/exp11_supervised|
| 20230116    | TransformerV4 | MagWarp            | 82.18%   | sl29 |  Parkland_TransformerV4/exp12_supervised|
| 20230116    | TransformerV4 | TimeWarp           | 90.95%   | sl29 |  Parkland_TransformerV4/exp13_supervised|
| 20230117    | TransformerV4 | Negation           | 86.06%   | sl29 |  Parkland_TransformerV4/exp14_supervised|
| 20230117    | TransformerV4 | Permutation        | 84.26%   | sl29 |  Parkland_TransformerV4/exp15_supervised|

#### 2. Vehicle classification 

|  Date       | Model |  Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |      :---: | :--: | :-- | :-- |
| 20230131    | DeepSense (Supervised)| 91.22%   | sl29 | Parkland_DeepSense/exp0_supervised_vehicle_classification_1.0 | use MixUp augmentation. |
| 20230131    | DeepSense + SimCLR | 89.89%   |  sl29 | Parkland_DeepSense/exp0_contrastive_SimCLR | use large dataset, batch size 256|
| 20230130    | TransformerV4 (Supervised)| 88.74%   |  sl29 | Parkland_TransformerV4/exp63_supervised_vehicle_classification_1.0 | use MixUp augmentation. |
| 20230130    | TransformerV4 + SimCLR | 93.03%   |  sl29 | Parkland_TransformerV4/exp22_contrastive_SimCLR | use large dataset, batch size 256|
| 20230130    | TransformerV4 + MoCoV3 | 91.03%   | tkimura4 | Parkland_TransformerV4/exp10_contrastive | use MoCo, batch size 64|

#### 3. Speed classification 

|  Date       | Model |  Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |      :---: | :--: | :-- | :-- |
| 20230131    | DeepSense (Supervised)| 59.37%   | sl29 | Parkland_DeepSense/exp0_supervised_speed_classification_1.0 | use MixUp augmentation. |
| 20230131    | DeepSense + SimCLR | 96.87%   |  sl29 | Parkland_DeepSense/exp0_contrastive_SimCLR | use large datasets, batch size 256|
| 20230131    | TransformerV4 (Supervised)| 56.25%   |  sl29 | Parkland_TransformerV4/exp0_supervised_speed_classification_1.0 | use MixUp augmentation. |
| 20230131    | TransformerV4 + SimCLR | 93.75%   |  sl29 | Parkland_TransformerV4/exp22_contrastive_SimCLR | use large datasets, batch size 256|

#### 4. Distance classification 

|  Date       | Model |  Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |      :---: | :--: | :-- | :-- |
| 20230131    | DeepSense (Supervised)| 54.54%   | sl29 | Parkland_DeepSense/exp1_supervised_distance_classification_1.0  | no augmentation. |
| 20230131    | DeepSense + SimCLR | 90.91%   |  sl29 | Parkland_DeepSense/exp0_contrastive_SimCLR | use large datasets, batch size 256|
| 20230131    | TransformerV4 (Supervised)| 54.54%   |  sl29 | Parkland_TransformerV4/exp1_supervised_distance_classification_1.0 | no augmentation. |
| 20230131    | TransformerV4 + SimCLR | 75.76%   |  sl29 | Parkland_TransformerV4/exp22_contrastive_SimCLR | use large datasets, batch size 256|


### ACIDS Results

#### 1. Vehicle Classification
|  Date       | Model |  Accuracy  | User | Weight Checkpoint   | Comment |
| :---:       |    :----:     |   :----:   |      :---: |  :--- | :--- |
| 20230206    | DeepSense (Supervised)     |  93.61%   | sl29 | ACIDS_DeepSense/exp5_supervised_vehicle_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230206    | DeepSense + SimCLR    |  73.57%   | sl29 | ACIDS_DeepSense/exp2_contrastive_SimCLR |  |
| 20230206    | TransformerV4 (Supervised) |  91.14%   | sl29 | ACIDS_TransformerV4/exp0_supervised_vehicle_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230206    | TransformerV4 + SimCLR |  88.91%   | sl29 | ACIDS_TransformerV4/exp0_contrastive_SimCLR |  |
| 20230207    | TransformerV4 + MoCo |  86.58%   | sl29 | ACIDS_TransformerV4/exp1_contrastive_MoCo |  |

#### 2. Terrain Classification
|  Date       | Model |  Accuracy  | User | Weight Checkpoint   | Comment |
| :---:       |    :----:     |   :----:   |      :---: |  :--- | :--- |
| 20230201    | DeepSense (Supervised)     |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_supervised_terrain_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | TransformerV4 (Supervised) |  90.76%   | sl29 | ACIDS_TransformerV4/exp0_supervised_terrain_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | DeepSense + SimCLR    |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_contrastive_SimCLR |  |
| 20230201    | TransformerV4 + SimCLR |  xx.xx%   | sl29 | ACIDS_TransformerV4/exp0_contrastive_SimCLR | |

#### 3. Distance Classification
|  Date       | Model |  Accuracy  | User | Weight Checkpoint   | Comment |
| :---:       |    :----:     |   :----:   |      :---: |  :--- | :--- |
| 20230201    | DeepSense (Supervised)     |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_supervised_speed_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | TransformerV4 (Supervised) |  xx.xx%  | sl29 | ACIDS_DeepSense/exp0_supervised_distance_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | DeepSense + SimCLR    |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_contrastive_SimCLR | no augmentations |
| 20230201    | TransformerV4 + SimCLR |  xx.xx%   | sl29 |  |  |

#### 4. Speed Classification
|  Date       | Model |  Accuracy  | User | Weight Checkpoint   | Comment |
| :---:       |    :----:     |   :----:   |      :---: |  :--- | :--- |
| 20230201    | DeepSense (Supervised)     |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_supervised_speed_classification_1.0 | no augmentations |
| 20230201    | TransformerV4 (Supervised) |  xx.xx%   | sl29 |  | |
| 20230201    | DeepSense + SimCLR    |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_contrastive_SimCLR | no augmentations |
| 20230201    | TransformerV4 + SimCLR |  xx.xx%   | sl29 |  | |
