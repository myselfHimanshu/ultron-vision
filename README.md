# ULTRON VISION MODELS

<p align="center">

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)&nbsp;&nbsp;&nbsp;[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)&nbsp;&nbsp;&nbsp;[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)&nbsp;&nbsp;&nbsp;[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

</p>

This repo contains deep learning vision models.

## BENCHMARKS

|Model|Data|Best Valid Accuracy|Epochs Used|Parameters|
|--|--|--|--|--|
|mnist_custom_model|MNIST|99.49%|20|7808|
|custom_atrous_model|CIFAR-10|93.02%|50|220,778|
|resnet-18|CIFAR-10|93.44%|50|11,173,962|
|resnet-18(albumentation)|CIFAR-10|92.17%|50|11,173,962|

## FUNCTIONS IMPLEMENTED

- [x] pytorch-transformation
- [x] albumentation-transformation
- [x] data loader
- [x] training
- [x] validation
- [x] predict single image
- [x] interpret prediction using gradcam on single image
- [x] logger
- [ ] custom dataset implementation
- [ ] custom loss function


## INSTALLATION

```
// install virtualenv
$ python3 -m pip install --user virtualenv

// create environment
$ python3 -m venv myenv

// activate environment
$ source myenv/bin/activate

// install dependencies from requirements.txt
$ pip install -r requirements.txt

// deactivate environment
$ deactivate
```

## FOLDER STRUCTURE

```
.
├── agents // define training and testing
│ ├── base.py
│ ├── mnist_agent.py
│ └── cifar10_agent.py
│
├── configs // store networks configuration parameters
│ ├── mnist_config.json
│ └── cifar10_config.json
│
├── data // raw, processed data + test images
│
├── experiments // store checkpoints, logs and outputs for experiment
│ └── cifar_10_exp
│   ├── logs // train validation score log
│   └── stats // plots, visualized images
│
├── infdata // initialize and fetch dataset
│ ├── dataset // defining custom dataset class
│ ├── transformation // custom transformation class
│ └── loader // data loader
│   └── cifar10_dl.py
│
├── logger.py // define the logger
├── losses // custom network losses
│
├── networks // define our network
│ ├── resnet_net.py
│ ├── mnist_net.py
│ └── utils.py
│
├── notebooks // jupyter notebooks for experiments
│ └── cifar10_nb.ipynb
│
├── utils // helper functions
├── README.md
├── requirements.txt
├── ultron.py
└── ultron.sh
```


## TRAINING ULTRON

In `ultron.sh` provide `ultron.py` and `path-to-config-file`.

```
// make ultron.sh executable file
$ chmod +x ultron.sh

// train ultron
$ ./ultron.sh
```

