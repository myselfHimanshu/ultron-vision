# ULTRON VISION API

<p align="center">

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)&nbsp;&nbsp;&nbsp;[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)&nbsp;&nbsp;&nbsp;[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)&nbsp;&nbsp;&nbsp;[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

</p>

This repo contains deep learning vision models.

AIM : build a generic computer vision model.

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
├── data // raw or processed data
│
├── experiments // store checkpoints, logs and outputs for experiment
│
├── infdata // initialize and fetch dataset
│ ├── dataset // defining custom dataset class
│ ├── transformation // custom transformation class
│ └── loader // data loader
│   └── mnist_dl.py
│
├── logger.py // define the logger
├── losses // custom network losses
│
├── networks // define our network
│ ├── cifar10_net.py
│ ├── mnist_net.py
│ └── utils.py
│
├── notebooks // jupyter notebooks for experiments
│ └── mnist_nb.ipynb
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

