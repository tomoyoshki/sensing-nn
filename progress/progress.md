# Weekly Progress

Tommy

## December 27th 2022

### Major modifications

```bash
src
├── data
│   └── Parkland.yaml # added TransformerV4 configurations
├── src
│   └── models
│       ├── SwinModules.py # created SwinModules module
│       └── TransformerV4.py # created TransformerV4 model
```

- Modified some files to comment out noise related requirements

### Usage

```
python3 train.py -gpu=0 -dataset=Parkland -stage=pretrain_classifier -model="TransformerV4"
```

### Results

- Achieving ~57% accuracy

### Problems

1. Minors

- Augmenter
    - `augmenter.train()` gives error saying such methods
- Arg parsers
    - Commented out save_emb, elastic_mod, and some noise related

2. Data dimension & Model hierarchy

- Seismic has `[64, 10, 20, 2] (batch, time_interval, spectrum, channels)`
- Audio has `[64, 10, 1600, 2] (batch, time_interval, spectrum, channels)`
    - Patch embedding requires `(batch, height, width, channel)` and performs partition and linear embedding
        - Should I use 10 * 20 as the height and width for seismic
        - And shoudl I use `[64, 10, 1600 / 80, 2 * 80]` and use 10 * 20 as the height and width for audio? 
        - Or should I have a different embedding method that uses 10 * 1600 as the height and width for audio?
        - Or should I **not use** these for the image dimension or the Swin Transformer for Real time data does not use image dimension at all? 
    - Since I am using 10 * 20 as the image dimension, and using 2 as the patch size, the hierarchy would be shallow, since (10 / 2) and (20 / 2) (first level) is the minimum it can go and (10 / 4) and (20 / 4) will give non-integer and odd integers which give dimension misalign error for patches
    - **Key questions: ** is there any other dimension that I could use for image dimension or should Swin transformer for time-series data be implemented differently without an image dimension

3. Fusions and sensors

- Currently I have a patch embedding and a layer for Frequency & interval for frequency and interval parts.
- I also have another layer afterward for modality part. (Have not tried locations)
- I am not sure if TransformerFusion block could be used sicne it is MHA, while SwinTransformer blocks already have these attention modules. 
- I am not sure if the first layer for frequency & interval is enough in getting information about the two features, should I have two separate layers for frequency and interval?