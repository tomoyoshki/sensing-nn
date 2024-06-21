# FOCAL (VibroFM)

Major changes

1. Add multiclass, if you want to run multiclass, simply add `-finetune_tag=multiclass`
2. Add different finetuning set and testing set support
    - E.g. `-finetune_set=ictfiltered`, `-finetune_set=ictexclusive`
    - E.g. `-test_set=ictfiltered`, `-test_set=ictexclusive`
    - Dataset explanation
        - This selects the set with different index path listed in `Parkland.yaml`
        - ICT Filtered is the one where Yizhuo did the 15 meter filtering, (15 min for train and 45 min for val, non-filtered 45min  for test)
        - ICT Exclusive is the new one with <15 meter and >70 meter filtering, (15 min for train and 45 min for val, non-filtered 45min  for test)
3. Add mlp classifier support (not necessary)
4. Add vehicle detector support, so if "dual" is in finetune_tag, then the classifier returns both class predictions (which class this vehicle belongs to) and vehilcle detection (whether there is a vehicle), if this is on, then vehicles detected as negative (is not nearby) will be discarded, see TransformerV4_CMC.py and finetune.py and eval_functions.py to see the discard logic, not necessary for the paper but possibly useful for demo

Example usage finetuning

```bash
# For multiclass finetuning
python3 train.py -gpu=X -model=TransformerV4 -learn_framework=CMCV2 -stage=finetune -finetune_set=ictfiltered -finetune_tag=multiclass

# For multiclass and vehicle detection
python3 train.py -gpu=X -model=TransformerV4 -learn_framework=CMCV2 -stage=finetune -finetune_set=ictfiltered -finetune_tag=multiclass_dual

# Different set
python3 train.py -gpu=X -model=TransformerV4 -learn_framework=CMCV2 -stage=finetune -finetune_set=ictexclusive -finetune_tag=multiclass_dual
```

**Weights**

```
All on Eugene

pretrain weights: /home/tkimura4/FTSSense/weights/Parkland_TransformerV4_debug/exp100_contrastive_CMCV2/Parkland_TransformerV4_pretrain_latest.pt

Finetuned weights (multi class on ict exclusive < 15 meters & > 70 meters): /home/tkimura4/FTSSense/weights/Parkland_TransformerV4_debug/exp100_contrastive_CMCV2/Parkland_TransformerV4_vehicle_classification_finetune_ictexclusive_1.0_multiclass_best.pt

Finetuned weights (multi class on ict filtered - < 15 meters): /home/tkimura4/FTSSense/weights/Parkland_TransformerV4_debug/exp100_contrastive_CMCV2/Parkland_TransformerV4_vehicle_classification_finetune_ictfiltered_1.0_multiclass_best.pt
```

**Note that for testing, simply change `train.py` to `test.py` and ensure everything else is the same, and add -test_set=XXX**