"""
Base Agent Class
"""

import logging

class BaseAgent:
    """
    base functions to be overloaded by other defined agents
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def load_checkpoint(self, file_name):
        """
        Load checkpoint loader
        :param file_name: name of checkpoint file path
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint Saver
        :param file_name: name of checkpoint file path
        :param is_best: boolean flag indicating current metrix is best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training iteration
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model evaluation
        :return:
        """
        raise NotImplementedError