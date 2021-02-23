# AnomalyDetection Pytorch

Fork from - https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch
Pytorch version of - https://github.com/WaqasSultani/AnomalyDetectionCVPR2018

## Known Issues:


## Install Anaconda Environment
```conda env create -f environment.yml```

```conda activate ad```


## Features Extraction
```python feature_extractor.py --dataset_path "path-to-dataset"  --pretrained_3d "path-to-pretrained-c3d"```

## Training
```python TrainingAnomalyDetector_public.py --features_path "path-to-dataset" --annotation_path "path-to-train-annos"```
