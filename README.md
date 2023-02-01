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


### Augmenter performance on Parkland dataset with supervised train
|  Date       | Model | Augmenter   |  Accuracy  | Weight Checkpoint |
| :---:       |    :----:    |    :----:   |      :---: |       :---: | 
| 20230116    | TransformerV4 | PhaseShift         | 88.68%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp6_supervised |
| 20230116    | TransformerV4 | FreqMask           | 85.53%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp9_supervised |
| 20230116    | TransformerV4 | TimeMask           | 72.94%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp7_supervised |
| 20230116    | TransformerV4 | HorizontalFlip     | 84.59%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp10_supervised|
| 20230116    | TransformerV4 | Jitter             | 84.46%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp11_supervised|
| 20230116    | TransformerV4 | MagWarp            | 82.18%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp12_supervised|
| 20230116    | TransformerV4 | TimeWarp           | 90.95%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp13_supervised|
| 20230117    | TransformerV4 | Negation           | 86.06%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp14_supervised|
| 20230117    | TransformerV4 | Permutation        | 84.26%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp15_supervised|

### Vehicle classification performance on Parkland dataset

|  Date       | Model |  Accuracy  | Weight | Comment | 
| :---:       |    :----:    |      :---: | :--: | :--: |
| 20230131    | DeepSense (Supervised)| 91.22%   | /home/sl29/FoundationSense/weights/Parkland_DeepSense/exp0_supervised_vehicle_classification_1.0 | use MixUp augmentation. |
| 20230131    | DeepSense + SimCLR | 89.89%   | /home/sl29/FoundationSense/weights/Parkland_DeepSense/exp0_contrastive_SimCLR | use large dataset, batch size 256|
| 20230130    | TransformerV4 (Supervised)| 88.74%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp63_supervised_vehicle_classification_1.0 | use MixUp augmentation. |
| 20230130    | TransformerV4 + SimCLR | 93.03%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp22_contrastive_SimCLR | use large dataset, batch size 256|
| 20230130    | TransformerV4 + MoCoV3 | 91.03%   | /home/tkimura4/FoundationSense/weights/Parkland_TransformerV4/exp10_contrastive | use MoCo, batch size 64|

### Speed classification performance on Parkland dataset

|  Date       | Model |  Accuracy  | Weight | Comment | 
| :---:       |    :----:    |      :---: | :--: | :--: |
| 20230131    | DeepSense (Supervised)| 59.37%   | /home/sl29/FoundationSense/weights/Parkland_DeepSense/exp0_supervised_speed_classification_1.0 | use MixUp augmentation. |
| 20230131    | DeepSense + SimCLR | 96.87%   | /home/sl29/FoundationSense/weights/Parkland_DeepSense/exp0_contrastive_SimCLR | use large datasets, batch size 256|
| 20230131    | TransformerV4 (Supervised)| 56.25%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp0_supervised_speed_classification_1.0 | use MixUp augmentation. |
| 20230131    | TransformerV4 + SimCLR | 93.75%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp22_contrastive_SimCLR | use large datasets, batch size 256|

### Distance classification performance on Parkland dataset

|  Date       | Model |  Accuracy  | Weight | Comment | 
| :---:       |    :----:    |      :---: | :--: | :--: |
| 20230131    | DeepSense (Supervised)| 54.54%   |/home/sl29/FoundationSense/weights/Parkland_DeepSense/exp1_supervised_distance_classification_1.0  | no augmentation. |
| 20230131    | DeepSense + SimCLR | 90.91%   | /home/sl29/FoundationSense/weights/Parkland_DeepSense/exp0_contrastive_SimCLR | use large datasets, batch size 256|
| 20230131    | TransformerV4 (Supervised)| 54.54%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp1_supervised_distance_classification_1.0 | no augmentation. |
| 20230131    | TransformerV4 + SimCLR | 75.76%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp22_contrastive_SimCLR | use large datasets, batch size 256|


### Model performance on ACIDS dataset

|  Date       | Model |  Accuracy  | Weight Checkpoint   |
| :---:       |    :----:     |   :----:   |      :---: |
| 20230130    | DeepSense (Supervised)     |  93.44%   | /home/sl29/FoundationSense/weights/ACIDS_DeepSense/exp22_supervised_vehicle_classification_1.0          |
| 20230130    | TransformerV4 (Supervised) |  88.67%   | /home/sl29/FoundationSense/weights/ACIDS_TransformerV4/exp20_supervised_vehicle_classification_1.0       |
