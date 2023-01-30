# FoundationSense

Transformer-based foundation models for (multi-modal) time-series sensing data


### Training 

#### Supervised learning

```
CUDA_VISIBLE_DEVICES=0 python3 train.py -gpu=0 -dataset=Parkland -train_mode=supervised -model=TransformerV4
```

#### Self supervised pretraining (contrastive)

```
CUDA_VISIBLE_DEVICES=0 python3 train.py -gpu=0 -dataset=Parkland -train_mode=contrastive -contrastive_framework=SimCLR -stage=pretrain -model=TransformerV4
```

#### Self supervised finetuning (contrastive)

```
CUDA_VISIBLE_DEVICES=0 python3 train.py -gpu=0 -dataset=Parkland -train_mode=contrastive -contrastive_framework=SimCLR -stage=finetune -model=TransformerV4
```

### Testing 

```
CUDA_VISIBLE_DEVICES=0 python3 test.py -gpu=0 -dataset=Parkland -train_mode=supervised -model=TransformerV4 -model_weight=/home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp10_supervised
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

### Model performance on Parkland dataset

|  Date       | Model |  Accuracy  | Weight | Comment | 
| :---:       |    :----:    |      :---: | :--: | :--: |
| 20230130    | TransformerV4 (Supervised)| 90.09%   | | |
| 20230130    | TransformerV4 + SimCLR | 93.03%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp22_contrastive | use large datasets, batch size 256|

### Model performance on ACIDS dataset

|  Date       | Model |  Accuracy  | Weight Checkpoint   |
| :---:       |    :----:     |   :----:   |      :---: |
| 20230130    | DeepSense (Supervised)     |  93.48%   | /home/sl29/FoundationSense/weights/ACIDS_DeepSense/exp21_supervised           |
| 20230130    | TransformerV4 (Supervised) |  89.71%   | /home/sl29/FoundationSense/weights/ACIDS_TransformerV4/exp17_supervised       |
