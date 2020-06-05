# ULTRON VISION MODELS

<p align="center">

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)&nbsp;&nbsp;&nbsp;[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)&nbsp;&nbsp;&nbsp;[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)&nbsp;&nbsp;&nbsp;[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

</p>

This repo contains deep learning vision models from scratch. This is work in progress!!! 

## BENCHMARKS 

I'll be continously updating the scores. Experiment results can be found in `experiments` folder (for details see Folder Structure below).

|Model|Data|Best Validation Accuracy|Total Epochs|Parameters|Experiment Result|
|--|--|--|--|--|--|
|mnist_custom_model|MNIST|99.49%|20|7,808|mnist_exp_01|
|custom_atrous_model|CIFAR-10|85.91%|20|220,778|cifar10_exp_01_atrous|
|resnet-18|CIFAR-10|93.44%|50|11,173,962|cifar10_exp_03_resnet|
|resnet-18(album)|CIFAR-10|92.17%|50|11,173,962|cifar10_exp_04_resnet_album|
|resnet-18(album/lr-finder)|CIFAR-10|89.80%|50|11,173,962|cifar10_exp_06_resnet_album_findlr|


## HARDWARE CONFIGURATION

- GPUs : NVIDIA® GeForce® GTX 1080Ti
- GPU count : 1
- vCPUs : 4
- Memory : 12 GiB
- Disk : 80 GiB
- [Genesis Cloud](https://gnsiscld.co/496pv5j) offers GPU cloud computing at unbeatable cost efficiency.


## FUNCTIONS IMPLEMENTED

- [x] pytorch-transformation
- [x] [albumentation-transformation](https://albumentations.readthedocs.io/en/latest/index.html)
- [x] data loader
- [x] training
- [x] validation
- [x] predict single image
- [x] learning rate finder with reduce-lr-on-plateau
- [ ] cyclic-lr
- [x] interpret misclassified images using gradcam
- [x] logging
- [x] save model checkpoint
- [x] load model checkpoint
- [ ] custom dataset implementation
- [ ] custom loss function
- [ ] using [weights and biases](https://www.wandb.com/) for logging experiments


## INSTALLATION

```bash
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

```java
.
├── agents // define training and validation
│   ├── base.py
│   ├── mnist_agent.py
│   └── cifar10_agent.py
│
├── configs // store networks configuration parameters
│   ├── mnist_config.json
│   └── cifar10_config.json
│
├── data // raw, processed data + test images
│
├── experiments // store checkpoints, logs and outputs for experiment
│   └── cifar10_exp*
│       ├── logs // agent logs
│       ├── stats // training validation scores, plots and images visualization data
│       └── summaries // experiment config file used and network architecture
│
├── infdata // initialize and fetch dataset
│   ├── dataset // defining custom dataset class
│   ├── transformation // custom transformation class
│   └── loader // data loader
│       └── cifar10_dl.py
│
├── inference // define inference agent
│   ├── base.py
│   └── cifar_iagent.py
│
├── logger.py // define the logger
├── losses // custom network losses
│
├── networks // define our network
│   ├── resnet_net.py
│   ├── mnist_net.py
│   └── utils.py
│
├── notebooks // jupyter notebooks for experiments
│   └── cifar10_nb.ipynb
│
├── utils // helper functions
├── README.md
├── requirements.txt
├── ultron.py
└── ultron.sh
```

## TRAINING ULTRON

In `ultron.sh` provide `ultron.py` and `path-to-config-file`.

```bash
// make ultron.sh executable file
$ chmod +x ultron.sh

// train ultron
$ ./ultron.sh
```
