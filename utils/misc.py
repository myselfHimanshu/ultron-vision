import logging
import torch

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

