# ULTRON VISION API

This repo contains vision deep learning models.

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





