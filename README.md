# ULTRON VISION API

This repo contains vision deep learning models.

AIM : build a generic computer vision model.

## INSTALLATION

```
```

## FOLDER STRUCTURE

```
.
├── agents // define training and testing
│ ├── base.py
│ └── cifar10_agent.py
├── checkpoints // store trained models
├── configs // store networks configuration parameters
│ └── cifar10_config.json
├── data // raw or processed data
├── infdata // initialize and fetch dataset
│ ├── dataset // defining custom dataset class
│ ├── transformation // custom transformation class
│ └── loader // data loader
│   └── cifar10_dl.py
├── logger.py // define the logger
├── losses // custom network losses
├── networks // define our network
│ ├── cifar10_net.py
│ └── utils.py
├── notebooks // jupyter notebooks for experiments
│ └── cifar10_nb.ipynb
├── README.md
├── requirements.txt
├── ultron.py
└── utils.py
```





