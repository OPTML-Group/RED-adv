DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
    "mnist": 10,
}

DATASET_MEAN = {
    "cifar10": [125.307/255, 122.961/255, 113.8575/255],
    "cifar100": [125.307/255, 122.961/255, 113.8575/255],
    "tinyimagenet": [0.4802, 0.4481, 0.3975],
    "mnist": [0]
}

DATASET_STD = {
    "cifar10": [51.5865/255, 50.847/255, 51.255/255],
    "cifar100": [51.5865/255, 50.847/255, 51.255/255],
    "tinyimagenet": [0.2302, 0.2265, 0.2262],
    "mnist": [1]
}

TRAINING_ARGS = {
    "cifar10": "",
    "cifar100": "",
    "tinyimagenet": " --epochs 100 --save-freq 25",
    "mnist": " --resnet18 --epochs 3 --save-freq 1",
}