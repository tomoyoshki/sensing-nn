# FoundationSense

Transformer-based foundation models for (multi-modal) time-series sensing data


### Training 
```
python3 train.py -gpu=0 -dataset=Parkland -train_mode=supervised -model=TransformerV4
```

### Testing 
```
python3 test.py -gpu=0 -dataset=Parkland -train_mode=supervised -model=TransformerV4
```

### Model performance on Parkland dataset
|  Date       | Model | Branch:Commit   |  Accuracy  |
| :---:       |    :----:    |    :----:   |      :---: |
| 20221230    | TransformerV4 | main:c7f5b7c4d31c76ef42e57f5c62ed81d3d870435d           | 74.01%   |
| 20230104    | TransformerV4 | main:41d6036a5bbb76dd22df2c49687b854aa1955eec           | 78.23%   |
