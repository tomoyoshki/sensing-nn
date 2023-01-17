# FoundationSense

Transformer-based foundation models for (multi-modal) time-series sensing data


### Training 

#### Supervised learning

```
CUDA_VISIBLE_DEVICES=0 python3 train.py -gpu=0 -dataset=Parkland -train_mode=supervised -model=TransformerV4
```

#### Self supervised learning (contrastive)

```
python3 train.py -gpu=0 -dataset=Parkland -train_mode=contrastive -model=TransformerV4
```

### Testing 

```
CUDA_VISIBLE_DEVICES=0 python3 test.py -gpu=0 -dataset=Parkland -train_mode=supervised -model=TransformerV4 -model_weight=/home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp10_supervised
```

### Model performance on Parkland dataset

|  Date       | Model | Branch:Commit   |  Accuracy  |
| :---:       |    :----:    |    :----:   |      :---: |
| 20221230    | TransformerV4 | main:c7f5b7c4d31c76ef42e57f5c62ed81d3d870435d           | 74.01%   |
| 20230104    | TransformerV4 | main:41d6036a5bbb76dd22df2c49687b854aa1955eec           | 78.23%   |
| 20230105    | TransformerV4 | main:34b1b024a4b3ba83850dc5c19016ac1a8ca01c7e           | 83.47%   |
| 20230105    | TransformerV4 | main:5a9ded0aee37589755a591cf5a079de5549aa0fa           | 90.09%   |

### Augmenter performance on Parkland dataset with supervised train
|  Date       | Model | Augmenter   |  Accuracy  | Weight Checkpoint |
| :---:       |    :----:    |    :----:   |      :---: |       :---: | 
| 20230116    | TransformerV4 | PhaseShift         | 88.68%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp6_supervised |
| 20230116    | TransformerV4 | FreqMask           | 85.53%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp9_supervised |
| 20230116    | TransformerV4 | TimeMask           | 72.94%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp7_supervised |
| 20230116    | TransformerV4 | HorizontalFlip     | 84.59%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp10_supervised|
| 20230116    | TransformerV4 | Jitter             | 84.46%   | /home/sl29/FoundationSense/weights/Parkland_TransformerV4/exp11_supervised|
