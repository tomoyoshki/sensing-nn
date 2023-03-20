# FoundationSense

Transformer-based foundation models for (multi-modal) time-series sensing data

### [Internal] Complete Result
[Google sheet](https://docs.google.com/spreadsheets/d/1r2cNw9Dhy9kzQ601ogCgEJn-meDGkahuqeCOut9oUQs/edit?usp=sharing)


### Supervised Learning 

#### Training
```
python3 train.py -gpu=0 -dataset=Parkland -task=vehicle_classification -model=TransformerV4 -batch_size=64
```

#### Testing 
```
python3 test.py -gpu=0 -dataset=Parkland -task=vehicle_classification -model=TransformerV4 -model_weight=/home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp10_supervised
```

### Contrastive/Predictive/Generative Learning

#### Pretraining
```
python3 train.py -gpu=0 -dataset=Parkland -learn_framework=SimCLR -stage=pretrain -model=TransformerV4
```

#### Finetuning
```
python3 train.py -gpu=0 -dataset=Parkland -learn_framework=SimCLR -stage=finetune -task=vehicle_classification -model=TransformerV4
```

#### Testing 
```
python3 test.py -gpu=0 -dataset=Parkland -learn_framework=SimCLR -stage=finetune -task=vehicle_classification -model=TransformerV4 -model_weight=/home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp22_contrastive
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
| 20230116    | TransformerV4 | TimeWarp           | 9095%   | sl29 |  Parkland_TransformerV4/exp13_supervised|
| 20230117    | TransformerV4 | Negation           | 86.06%   | sl29 |  Parkland_TransformerV4/exp14_supervised|
| 20230117    | TransformerV4 | Permutation        | 84.26%   | sl29 |  Parkland_TransformerV4/exp15_supervised|

#### 2. Vehicle classification 

|  Date       | Model | Framework |   Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |   :----:    |    :---: | :--: | :-- | :-- |
| 20230131    | DeepSense| Supervised | 89.08%   | sl29 | Parkland_DeepSense/exp1_supervised_vehicle_classification_1.0 | mixup. |
| 20230131    | DeepSense | SimCLR | 89.89%   |  sl29 | Parkland_DeepSense/exp0_contrastive_SimCLR | use large dataset, batch size 256|
| 20230211    | DeepSense | MoCo | 88.34%   |  tkimura4 | Parkland_TransformerV4/exp0_contrastive_MoCo | use large dataset, batch size 64| 
| 20230218    | DeepSense | Cosmo | 91.36%   | sl29 | Parkland_DeepSense/exp0_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230220    | DeepSense | CMC | 94.58%   |  tkimura4 | Parkland_DeepSense/exp4_contrastive_CMC | use large datasets, batch size 128, infoNCE loss|
| 20230226    | DeepSense | MTSS | 46.95% | sl29 | | 
| 20230228    | DeepSense | ModPred | 42.86% | sl29 | | ||
| 20230317    | DeepSense | MAE | 52.38% | sl29 | | ||
| 20230130    | TransformerV4 | Supervised | 91.90%   |  sl29 | Parkland_TransformerV4/exp5_supervised_vehicle_classification_1.0 | mixup + phase_shift. |
| 20230130    | TransformerV4 | SimCLR | 93.03%   |  sl29 | Parkland_TransformerV4/exp22_contrastive_SimCLR | use large dataset, batch size 256|
| 20230212    | TransformerV4 | MoCo | 94.31%   |  tkimura4 | Parkland_TransformerV4/exp0_contrastive_MoCo | use large dataset, batch size 256|
| 20230223    | TransformerV4 | Cosmo | 23.04%   | sl29 | Parkland_TransformerV4/exp0_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230220    | TransformerV4 | CMC | 91.96%   |  tkimura4 | Parkland_TransformerV4/exp28_contrastive_CMC | use large dataset, batch size 128, infoNCE loss|
| 20230226    | TransformerV4 | MTSS | 42.19% | sl29 | | |
| 20230228    | TransformerV4 | ModPred | 43.47% | sl29 | | |
| 20230317    | TransformerV4 | MAE | 78.46% | sl29 | | ||

#### 3. Speed classification 

|  Date       | Model | Framework |   Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |  :---: |  :---: | :--: | :-- | :-- |
| 20230131    | DeepSense | Supervised | 84.37%   | sl29 | Parkland_DeepSense/exp0_supervised_speed_classification_1.0 | use MixUp augmentation. |
| 20230131    | DeepSense | SimCLR | 96.87%   |  sl29 | Parkland_DeepSense/exp0_contrastive_SimCLR | use large datasets, batch size 256|
| 20230211    | DeepSense | MoCo | 96.87%   |  tkimura4 | Parkland_TransformerV4/exp0_contrastive_MoCo | use large dataset, batch size 64|
| 20230220    | DeepSense | CMC | 78.125%   |  tkimura4 | Parkland_DeepSense/exp4_contrastive_CMC | use large datasets, batch size 128, infoNCE loss|
| 20230224    | DeepSense | Cosmo | 96.87%   | sl29 | Parkland_DeepSense/exp0_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230131    | TransformerV4 | Supervised | 68.75%   |  sl29 | Parkland_TransformerV4/exp0_supervised_speed_classification_1.0 | use MixUp augmentation. |
| 20230131    | TransformerV4 | SimCLR | 93.75%   |  sl29 | Parkland_TransformerV4/exp22_contrastive_SimCLR | use large datasets, batch size 256|
| 20230212    | TransformerV4 | MoCo | 93.60%   |  tkimura4 | Parkland_TransformerV4/exp0_contrastive_MoCo | use large datasets, batch size 64|
| 20230220    | TransformerV4 | CMC | 90.63%   |  tkimura4 | Parkland_TransformerV4/exp28_contrastive_CMC | use large datasets, batch size 128, infoNCE loss
| 20230224    | TransformerV4 | Cosmo | 40.62%   | sl29 | Parkland_TransformerV4/exp0_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |

#### 4. Distance classification 

|  Date       | Model | Framework |   Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |  :---: |  :---: | :--: | :-- | :-- |
| 20230131    | DeepSense | Supervised | 66.67%   | sl29 | Parkland_DeepSense/exp0_supervised_distance_classification_1.0  | mixup. |
| 20230131    | DeepSense | SimCLR | 90.91%   |  sl29 | Parkland_DeepSense/exp0_contrastive_SimCLR | use large datasets, batch size 256|
| 20230211    | DeepSense | MoCo | 84.84%   |  tkimura4 | Parkland_TransformerV4/exp0_contrastive_MoCo | use large dataset, batch size 64| 
| 20230220    | DeepSense | CMC | 78.79%   |  tkimura4 | Parkland_DeepSense/exp4_contrastive_CMC | use large datasets, batch size 128, infoNCE loss|
| 20230224    | DeepSense | Cosmo | 90.91%   | sl29 | Parkland_DeepSense/exp0_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230131    | TransformerV4 | Supervised | 54.54%   |  sl29 | Parkland_TransformerV4/exp1_supervised_distance_classification_1.0 | no augmentation. |
| 20230131    | TransformerV4 | SimCLR | 75.76%   |  sl29 | Parkland_TransformerV4/exp22_contrastive_SimCLR | use large datasets, batch size 256|
| 20230212    | TransformerV4 | MoCo | 76.01%   |  tkimura4 | Parkland_TransformerV4/exp0_contrastive_MoCo | use large datasets, batch size 64|
| 20230220    | TransformerV4 | CMC | 90.91%   |  tkimura4 | Parkland_TransformerV4/exp28_contrastive_CMC | use large datasets, batch size 128, infoNCE loss|
| 20230224    | TransformerV4 | Cosmo | 54.55%   | sl29 | Parkland_TransformerV4/exp0_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |


### ACIDS Results

#### 1. Vehicle Classification
|  Date       | Model | Framework |   Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |  :---: |  :---: | :--: | :-- | :-- |
| 20230206    | DeepSense | Supervised |  93.61%   | sl29 | ACIDS_DeepSense/exp5_supervised_vehicle_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230206    | DeepSense   | SimCLR | 77.12%   | sl29 | ACIDS_DeepSense/exp0_contrastive_SimCLR |  |
| 20230207    | DeepSense | MoCo | 76.62%   | sl29 | ACIDS_DeepSense/exp0_contrastive_MoCo | temperature 0.2 |
| 20230218    | DeepSense | Cosmo | 92.62%   | sl29 | ACIDS_DeepSense/exp13_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230221    | DeepSense | CMC | 89.04%   | tkimura4 | ACIDS_DeepSense/exp3_contrastive_CMC | batch size 128 |
| 20230228    | DeepSense | MTSS | 43.28% | sl29 | | ||
| 20230228    | DeepSense | ModPred | 51.37% | sl29 | | ||
| 20230317    | DeepSense | MAE | 52.8% | sl29 | | ||
| 20230206    | TransformerV4 | Supervised | 91.14%   | sl29 | ACIDS_TransformerV4/exp0_supervised_vehicle_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230206    | TransformerV4 | SimCLR | 88.41%   | sl29 | ACIDS_TransformerV4/exp1_contrastive_SimCLR |  |
| 20230207    | TransformerV4 | MoCo | 87.67%   | sl29 | ACIDS_TransformerV4/exp2_contrastive_MoCo | temperature 0.2 |
| 20230216    | TransformerV4 | Cosmo | 72.85%   | sl29 | ACIDS_TransformerV4/exp2_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230221    | TransformerV4 | CMC | 84.80%   | tkimura4 | ACIDS_TransformerV4/exp10_contrastive_CMC | batch size 128 |
| 20230228    | TransformerV4 | MTSS | 43.65% | sl29 | | ||
| 20230228    | TransformerV4 | ModPred | 51.68% | sl29 | | ||
| 20230317    | TransformerV5 | MAE | 85.62% | sl29 | | ||

#### 2. Terrain Classification
|  Date       | Model | Framework |   Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |  :---: |  :---: | :--: | :-- | :-- |
| 20230201    | DeepSense     |  Supervised | 95.61%   | sl29 | ACIDS_DeepSense/exp0_supervised_terrain_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | DeepSense    | SimCLR | 80.03%   | sl29 | ACIDS_DeepSense/exp0_contrastive_SimCLR |  |
| 20230224    | DeepSense | MoCo | 78.89%   | sl29 | ACIDS_DeepSense/exp0_contrastive_MoCo | temperature 0.2 |
| 20230221    | DeepSense | CMC | 93.84%   | tkimura4 | ACIDS_DeepSense/exp3_contrastive_CMC | batch size 128 |
| 20230224    | DeepSense | Cosmo | 90.62%   | sl29 | ACIDS_DeepSense/exp13_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230201    | TransformerV4 | Supervised | 94.04%   | sl29 | ACIDS_TransformerV4/exp0_supervised_terrain_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | TransformerV4 | SimCLR | 90.36%   | sl29 | ACIDS_TransformerV4/exp1_contrastive_SimCLR | |
| 20230224    | TransformerV4 | MoCo | 87.28%   | sl29 | ACIDS_TransformerV4/exp2_contrastive_MoCo | temperature 0.2 |
| 20230221    | TransformerV4 | CMC | 96.40%   | tkimura4 | ACIDS_TransformerV4/exp10_contrastive_CMC | batch size 128 |
| 20230224    | TransformerV4 | Cosmo | 94.62%   | sl29 | ACIDS_TransformerV4/exp2_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |

#### 3. Speed Classification
|  Date       | Model | Framework |   Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |  :---: |  :---: | :--: | :-- | :-- |
| 20230201    | DeepSense | Supervised |  65.74%   | sl29 | ACIDS_DeepSense/exp0_supervised_speed_classification_1.0 | no augmentations |
| 20230201    | DeepSense    | SimCLR | 20.75%   | sl29 | ACIDS_DeepSense/exp0_contrastive_SimCLR | no augmentations |
| 20230224    | DeepSense | MoCo | 21.86%   | sl29 | ACIDS_DeepSense/exp0_contrastive_MoCo | temperature 0.2 |
| 20230221    | DeepSense | CMC | 25.70%   | tkimura4 | ACIDS_DeepSense/exp3_contrastive_CMC | batch size 128 |
| 20230224    | DeepSense | Cosmo | 32.17%   | sl29 | ACIDS_DeepSense/exp13_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230201    | TransformerV4 | Supervised | 53.83%   | sl29 | ACIDS_TransformerV4/exp0_supervised_speed_classification_1.0 | |
| 20230201    | TransformerV4 | SimCLR | 17.06%   | sl29 | ACIDS_TransformerV4/exp1_contrastive_SimCLR | |
| 20230224    | TransformerV4 | MoCo | 23.25%   | sl29 | ACIDS_TransformerV4/exp2_contrastive_MoCo | temperature 0.2 |
| 20230221    | TransformerV4 | CMC | 44.92%   | tkimura4 | ACIDS_TransformerV4/exp10_contrastive_CMC | batch size 128 |
| 20230224    | TransformerV4 | Cosmo | 23.39%   | sl29 | ACIDS_TransformerV4/exp2_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |

<!-- #### 4. Distance Classification (TODO: Data labels are not ready yet)
|  Date       | Model | Framework |   Accuracy  | User | Weight | Comment | 
| :---:       |    :----:    |  :---: |  :---: | :--: | :-- | :-- |
| 20230201    | DeepSense | Supervised     |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_supervised_speed_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | DeepSense | SimCLR |  xx.xx%   | sl29 | ACIDS_DeepSense/exp0_contrastive_SimCLR | no augmentations |
| 20230224    | DeepSense | MoCo | xx.xx%   | sl29 | ACIDS_DeepSense/exp1_contrastive_MoCo | temperature 0.2 |
| 20230221    | DeepSense | CMC | 75.69%   | tkimura4 | ACIDS_DeepSense/exp3_contrastive_CMC | batch size 128 |
| 20230224    | DeepSense | Cosmo | xx.xx%   | sl29 | ACIDS_DeepSense/exp13_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 |
| 20230201    | TransformerV4 | Supervised |  xx.xx%  | sl29 | ACIDS_DeepSense/exp0_supervised_distance_classification_1.0 | channel_shuffle, mixup, phase_shift |
| 20230201    | TransformerV4 | SimCLR | xx.xx%   | sl29 |  |  |
| 20230224    | TransformerV4 | MoCo | xx.xx%   | sl29 | ACIDS_TransformerV4/exp1_contrastive_MoCo | temperature 0.2 |
| 20230221    | TransformerV4 | CMC | 85.85%   | tkimura4 | ACIDS_TransformerV4/exp10_contrastive_CMC | batch size 128 |
| 20230224    | TransformerV4 | Cosmo | xx.xx%   | sl29 | ACIDS_TransformerV4/exp2_contrastive_Cosmo | temperature 1, no augmentation, batch size 256 | -->
