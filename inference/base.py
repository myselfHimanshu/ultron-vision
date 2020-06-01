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

    def predict(self):
        """
        Prediction on test dataset
        :return:
        """
        raise NotImplementedError

    def plot_accuracy_graph(self):
        """
        Plot accuracy graph for train and valid dataset
        :return:
        """
        raise NotImplementedError

    def plot_loss_graph(self):
        """
        Plot loss graph for train and valid dataset
        :return:
        """
        raise NotImplementedError

    def show_misclassified_images(self, n=25):
        """
        Show misclassified images
        :return:
        """
        raise NotImplementedError

    def interpret_images(self, image_data, image_label):
        """
        Grad Cam for interpreting and prediting class of image
        """
        raise NotImplementedError