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
python run.py --stage 1 --thread 3 --gpu 0,1,2,3,4,5,6,7
```
TODO: split training and attacking stage.

### Get attack images
```bash
python run.py --stage 1 --thread 3 --gpu 0,1,2,3,4,5,6,7
```

### Aggregate datas
```bash
python grep_data.py # for each attack strength
python grep_class_data.py # for mixed datasets
```

### Train parsing model
```bash
python run.py --stage 2 --thread 3 --gpu 0,1,2,3,4,5,6,7
```

### Evaluate parsing model
```bash
python run.py --stage 3 --thread 20 --gpu 0,1,2,3,4,5,6,7
```