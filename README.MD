# RED-adv

## Prepairation

Change the path related arguments in [path.py](./global_args/path.py)

### TinyImageNet Preparation
```
cd TinyImageNet
bash run.sh
```

## Pipeline

### Train victim model
```bash
python run.py --stage 1 --thread 3 --gpus 0,1,2,3,4,5,6,7
```
TODO: split training and attacking stage.

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

### Optional arguments in run.py
```
--stage STAGE    To decide which part of commands to execute. (1, 2, 3)
--gpus GPUS      Run on which gpus. e.g.: --gpus 0,1,2,3
--thread THREAD  Number of commands running parallel in one gpu.
--debug          Generate commands without executing if tagged.
```