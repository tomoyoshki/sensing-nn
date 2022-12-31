# FoundationSense

Transformer-based foundation models for (multi-modal) time-series sensing data

```
python3 train.py -gpu=0 -dataset=Parkland -stage=pretrain_classifier -model=TransformerV4
```

## Model performance on Parkland dataset
|  Date       | Model | Branch:Commit   |  Accuracy  |
| :---:       |    :----:    |    :----:   |      :---: |
| 20221230    | TransformerV4 | main:c7f5b7c4d31c76ef42e57f5c62ed81d3d870435d           | 74.01%   |
