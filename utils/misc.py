import logging
import torch

import numpy as np
import matplotlib.pyplot as plt
import json

def imshow(img, std, mean, clip=False):
    """
    plot image
    :param img: Image tensor
    :param std: global standard deviation
    :param mean: global mean
    :retun:
    """

    image = img.detach().cpu().numpy()
    image = image.transpose((1, 2, 0))
    std = np.array(std)
    mean = np.array(mean)
    image = std*image + mean
    if clip:
        image = np.clip(image, 0, 1)
    plt.imshow((image*255).astype(np.uint8))

def visualize_data(images, std, mean, target=None, classes=None, n=30, path=None):
    """
    Visualize data from train loader

    :param images: list of images
    :param std: global standard deviation
    :param mean: global mean
    :param target: labels for respective images
    :param classes: labels to classes mapping
    :param n: top-n images to show
    :return:
    """
    figure = plt.figure(figsize=(10,10))

    for i in range(1, n+1):
        plt.subplot(5,n//5,i)
        plt.axis('off')
        imshow(images[i-1], std, mean)
        plt.title("Actual : {}".format(classes[target[i-1]]))

    plt.tight_layout()
    if path!=None:
        figure.savefig(path)

def regularize_loss(model, loss, decay, norm_value):
    """
    L1/L2 Regularization
    decay : l1/l2 decay value
    norm_value : the order of norm
    """
    r_loss = 0
    # get sum of norm of parameters
    for param in model.parameters():
        r_loss += torch.norm(param, norm_value)
    # update loss value
    loss += decay * r_loss

    return loss

def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")

    import sys
    from subprocess import call

    logger.info(f'__Python VERSION : {sys.version}')
    logger.info(f'__pytorch VERSION : {torch.__version__}')
    # logger.info(f'__CUDA VERSION')
    # call(["nvcc","--version"])
    logger.info(f'__CUDNN VERSION : {torch.backends.cudnn.version()}')
    logger.info(f'__Number CUDA Devices : {torch.cuda.device_count()}')
    logger.info('__Devices')
    call(["nvidia-smi","--format=csv","--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info(f'Active CUDA Device : GPU {torch.cuda.current_device()}')
    logger.info(f'Available devices {torch.cuda.device_count()}')
    logger.info(f'Current CUDA device {torch.cuda.current_device()}')

