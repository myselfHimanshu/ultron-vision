# ULTRON VISION MODELS

<p align="center">

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)&nbsp;&nbsp;&nbsp;[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)&nbsp;&nbsp;&nbsp;[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)&nbsp;&nbsp;&nbsp;[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

</p>

## WHAT IS THIS REPO ABOUT? 

Structured approach to learning and implementing the fundamentals of State of the art vision models. The models from built from scratch. 

Follow below for updates on how it has been built over time. (This repo is work in progress!!!)

## UPDATES AND RESULTS

> Codes are available as gist as jupyter notebooks and blogs are for first 3 categories, rest are WIP. Models in these jupter notebooks were trained on Google Colab.

<details>
    <summary>29th March 2020</summary>

- Machine Learning Intuition, Background & Basics
- Python 101 for Machine Learning
- [blog](https://myselfhimanshu.github.io/posts/cnn_01/)

</details>

<details>
    <summary>5th April 2020</summary>

- Convolutions, Pooling Operations & Channels
- Pytorch 101 for Vision Machine Learning
- [blog](https://myselfhimanshu.github.io/posts/cnn_02/)

</details>

<details>
    <summary>12th April 2020</summary>

- Kernels, Activations and Layers
- [blog](https://myselfhimanshu.github.io/posts/cnn_03/)

</details>

<details>
    <summary>19th April 2020</summary>

- Architectural Basics. Finding suitable model architecture for the objective
- MNIST model training 
    - parameters used 13,402
    - epochs=20
    - highest test accuracy = 99.46%, epoch = 19th
    - [notebook link](https://gist.github.com/myselfHimanshu/6a8b74689799aa31fab5c7406c435461)

</details>

<details>
    <summary>26th April 2020</summary>

- Receptive Field : core fundamental concept
- MNIST model training
    - parameters used 7808
    - epochs=15
    - highest test accuracy = 99.43%, epoch = 11th 
    - [notebook link](https://gist.github.com/myselfHimanshu/82443162b618885628bff4d8a100ed21)

</details>

<details>
    <summary>3rd May 2020</summary>

- BN, Kernels & Regularization
- Mathematics behind Batch Normalization, Kernel Initialization and Regularization
- MNIST model training
    - using L1/L2 regularization with BN/GBN
    - BN : batch normalization
    - GBN : ghost batch normalization
    - best model : BN with L2
        - parameters used 7808
        - epochs=25
        - highest test accuaracy = 99.54%, epoch = 21st
    - [notebook link](https://gist.github.com/myselfHimanshu/61fbda0a7a451b53d7a39ee9fc2d91e2)

</details>

<details>
    <summary>10th May 2020</summary>

- Advanced Convolution
- Depthwise, Pixel Shuffle, Dilated, Transpose Convolutions
- CIFAR-10 dataset
- Achieve an accuracy of greater than 80% on CIFAR-10 dataset
    - architecture to C1C2C3C40 (basically 3 MPs)
    - total params to be less than 1M
    - RF must be more than 44
    - one of the layers must use Depthwise Separable Convolution
    - one of the layers must use Dilated Convolution
    - use GAP
- Result
    - parameters used 220,778
    - epochs = 20
    - highest test acc = 85.55%
    - [notebook link](https://gist.github.com/myselfHimanshu/bd9a700c332d8a91a1ada399ce318670)

</details>

<details>
    <summary>17th May 2020</summary>

- Receptive Fields and Network Architectures : Resnet Architecture
- Achieve an accuracy of greater than 85% on CIFAR-10 dataset
    - architecture ResNet18
- Result
    - parameters : 11,173,962
    - epoch : 50
    - training acc : 98.65%
    - testing acc : 89.78%
    - [notebook link](https://gist.github.com/myselfHimanshu/7969fe685b507286657fdea74e449d91)

</details>

> From here on, the codes are available in this repo. Models are trained on below given hardware configurations.

<details>
    <summary>31st May 2020</summary>

- Data Augmentation using Albumentations
- DNN Interpretability, Class Activation Maps using grad-cam
- Achieve an accuracy of greater than 87% on CIFAR-10 dataset
    - architecture ResNet18
    - Move transformations to Albumentations. 
    - Implement GradCam function. 
- Result
    - parameters : 11,173,962
    - epoch : 50
    - testing acc : 92.17%
    - [work link](https://github.com/myselfHimanshu/ultron-vision/tree/master/experiments/cifar10_exp_04_resnet_album)

</details>

<details>
    <summary>7th June 2020</summary>

- Advanced Concepts : Optimizers, LR Schedules, LR Finder & Loss Functions
- Achieve an accuracy of greater than 88% on CIFAR-10 dataset
    - architecture ResNet18
    - Add CutOut augmentation
    - Implement LR Finder (for SGD, not for ADAM)
    - Implement ReduceLROnPlateau
- Result
    - parameters : 11,173,962
    - epoch : 50
    - testing acc : 89.80%
    - [work link](https://github.com/myselfHimanshu/ultron-vision/tree/master/experiments/cifar10_exp-06_resnet_album_findlr)

</details>

<details>
    <summary>14th June 2020</summary>

- Super Convergence
- Cyclic Learning Rates, One Cycle Policy
- Achieve an accuracy of greater than 90% on CIFAR-10 dataset
    - 3Layer-DenseNet
    - Implement One Cycle Policy
- Result
    - parameters : 6,573,130
    - epoch : 24
    - testing acc : 91.02%
    - [work link](https://github.com/myselfHimanshu/ultron-vision/tree/master/experiments/cifar10_session11-exp-002)

</details>

## HARDWARE CONFIGURATION

- GPUs : NVIDIA® GeForce® GTX 1080Ti
- GPU count : 1
- vCPUs : 4
- Memory : 12 GiB
- Disk : 80 GiB
- [Genesis Cloud](https://gnsiscld.co/496pv5j) offers GPU cloud computing at unbeatable cost efficiency.


## MODULES IMPLEMENTED

- [x] pytorch-transformation
- [x] [albumentation-transformation](https://albumentations.readthedocs.io/en/latest/index.html)
- [x] data loader
- [x] training
- [x] validation
- [x] predict single image
- [ ] reduce-lr-on-plateau
- [x] lr-range test, optim lr finder
- [x] one-cycle-policy
- [x] interpret misclassified images using grad-cam
- [x] plots of accuracy, loss, learning_rate graphs wrt iterations
- [x] logging functionality
- [x] loading and saving model checkpoints
- [x] custom configuration file for training model
- [ ] custom loss function
- [ ] using [weights and biases](https://www.wandb.com/) for logging experiments
- [ ] torchstat or torchprof, layer-by-layer profiling of Pytorch models
- [ ] Deployment using Flask, EC2 or AWS-Lambda


## USE-CASES MODELS IMPLEMENTED FOR

- [x] Image Classification
- [ ] Object Detection
- [ ] Object Segmentation
- [ ] GANs

## NETWORK ARCHITECTURES IMPLEMENTED

- [x] Custom Networks
- [x] ResNet
- [x] DenseNet

## DATASETS USED

- [x] MNIST
- [x] CIFAR10
- [ ] TinyImageNet

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
│   ├── lr_finder
│   └── gradcam
│
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
