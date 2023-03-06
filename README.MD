# RED-adv

## Prepairation

Change the path related arguments in [path.py](./global_args/path.py)

### TinyImageNet Preparation
```
cd TinyImageNet
bash run.sh
```

### Pretrained DnCNN download
The pretrained DnCNN model is in [GoogleDrive](https://drive.google.com/file/d/1xZCmviKXxULit0dEyekhXs32waEc00Qa/view).

## Pipeline

### Train victim model
```bash
python run.py --stage 0 --thread 3 --gpus 0,1,2,3,4,5,6,7
```

### Get attack images
```bash
python run.py --stage 1 --thread 3 --gpus 0,1,2,3,4,5,6,7
```

### Aggregate datas
```bash
python grep_data.py # for each attack strength
python grep_class_data.py # for mixed datasets
```

### Train parsing model
```bash
python run.py --stage 2 --thread 3 --gpus 0,1,2,3,4,5,6,7
```

### Evaluate parsing model
```bash
python run.py --stage 3 --thread 20 --gpus 0,1,2,3,4,5,6,7
```

### Train and Evaluate parsing model with denoiser
```bash
python run.py --stage 2 --thread 3 --gpus 0,1,2,3,4,5,6,7 --denoise
python run.py --stage 3 --thread 5 --gpus 0,1,2,3,4,5,6,7 --denoise
```

## Optional arguments in run.py
```
--stage STAGE    To decide which part of commands to execute. (1, 2, 3)
--gpus GPUS      Run on which gpus. e.g.: --gpus 0,1,2,3
--thread THREAD  Number of commands running parallel in one gpu.
--denoise        Using denoiser when training attribute models. 
--debug          Generate commands without executing if tagged.
```
