## **Indroduction**

 The codes for the work "SAN-Net: Learning Generalization to Unseen Sites for Stroke Lesion Segmentation with Self-Adaptive Normalization".
 
## **Dataset**

 1. The dataset we used is ATLAS v1.2[1]. Note that the T1-weighted MR images from 229 patients were through z-score--normalization.
 [1] Liew S L, Anglin J M, Banks N W, et al. A large, open source dataset of stroke anatomical brain images and manual lesion segmentations[J]. Scientific data, 2018, 5(1): 1-11.
 2. The dataset is firstly processed according to [this](https://github.com/Andrewsher/ATLAS-dataset-generate-h5file) to get train.h5 file.
 3. Then, the dataset is processed to get three .npy files.
    python zscore.py

## Environment
|Name|Version  |
|--|--|
|Python|3.7|
|pytorch|1.7.0|
|torch|1.8.0|
|numpy|1.21.5|
|pandas|1.3.5|

## Train/Test

 `python train.py`
 
 The model parameters (trained on all sites from ATLAS v1.2 except for Site 5) are available on [this](https://mega.nz/folder/695zzSjK#nO3PzNjDxJSF8E1hVrc-bA).


