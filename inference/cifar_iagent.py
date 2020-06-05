"""
Inference Agent for CIFAR10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

from tqdm import tqdm

from agents.base import BaseAgent
# from networks.cifar10_atrous_net import Cifar10AtrousNet as Net
from networks.resnet_net import ResNet18 as Net
from infdata.loader.cifar10_dl import DataLoader as dl

# utils function
from utils.misc import *
from utils.gradcam import GradCam
from utils.grad_misc import visualize_cam

from torchsummary import summary
import json
import matplotlib.pyplot as plt
import numpy as np

import os
curr_dir = os.path.dirname(__file__)

class Cifar10IAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("INFERENCE MODE ACTIVATED!!!")
        self.config = config
        self.use_cuda = self.config['use_cuda']

        # create network instance
        self.model = Net()

        # define data loader
        self.dataloader = dl(config=self.config)

        # intitalize classes
        self.classes = self.dataloader.classes
        self.testclasses = self.dataloader.testclasses
        self.id2classes = {i:y for i,y in enumerate(self.classes)}
        self.id2tclasses = {i:y for i,y in enumerate(self.testclasses)}

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=self.config['momentum'])

        if not self.use_cuda and torch.cuda.is_available():
            self.logger.info('WARNING : You have CUDA device, you should probably enable CUDA.')

        # set manual seed
        self.manual_seed = self.config['seed']
        if self.use_cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config['gpu_device'])
            self.logger.info("Program will RUN on ****GPU-CUDA****")
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device('cpu')
            self.logger.info("Program will RUN on ****CPU****")


        self.load_checkpoint(self.config['checkpoint_file'])
        self.stats_file_name = os.path.join(self.config["stats_dir"], self.config["model_stats_file"])

    def load_checkpoint(self, file_name):
        """
        Latest Checkpoint loader
        :param file_name: name of checkpoint file
        :return:
        """
        file_name = os.path.join(self.config["checkpoint_dir"], file_name)
        checkpoint = torch.load(file_name, map_location='cpu')

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_epoch = checkpoint["epoch"]
        self.misclassified = checkpoint['misclassified_data']

    def predict(self):
        """
        predict image class
        :param image_name: image file name
        :return (str): class name
        """
        try:
            self.model.to(self.device)
            self.model.eval()
            predictions = []
            trues = []

            for data, target in self.dataloader.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(int(output.argmax(dim=1, keepdim=True).cpu().numpy()[0][0]))
                trues.append(int(target.cpu().numpy()[0]))
                
            self.logger.info("Test Image trueValues : " + str([self.id2tclasses[true] for true in trues]))    
            self.logger.info("Test Image Prediction : " + str([self.id2classes[pred] for pred in predictions]))
        except Exception as e:
            self.logger.info("Test image prediction FAILED!!!")
            self.logger.info(e)

    def plot_accuracy_graph(self):
        """
        Plot accuracy graph for train and valid dataset
        :return:
        """
        with open(self.stats_file_name) as f:
            data = json.load(f)

        train_acc = data["train_acc"]
        valid_acc = data["valid_acc"]

        epoch_count = range(1, self.config["epochs"]+1)
        fig = plt.figure(figsize=(10,10))
        
        plt.plot(epoch_count, train_acc)
        plt.plot(epoch_count, valid_acc)
        plt.legend(["train_acc","valid_acc"])
        plt.xlabel('Epoch')
        plt.ylabel("Accuracy")
        # plt.show();

        fig.savefig(os.path.join(self.config["stats_dir"], 'accuracy.png'))
        self.logger.info("Accuracy Graph saved.")

    def plot_loss_graph(self):
        """
        Plot loss graph for train and valid dataset
        :return:
        """
        with open(self.stats_file_name) as f:
            data = json.load(f)

        train_loss = data["train_loss"]
        valid_loss = data["valid_loss"]

        epoch_count = range(1, self.config["epochs"]+1)
        fig = plt.figure(figsize=(10,10))
        
        plt.plot(epoch_count, train_loss)
        plt.plot(epoch_count, valid_loss)
        plt.legend(["train_loss","valid_loss"])
        plt.xlabel('Epoch')
        plt.ylabel("Loss")
        # plt.show();

        fig.savefig(os.path.join(self.config["stats_dir"], 'loss.png'))
        self.logger.info("Loss Graph saved.")

    def show_misclassified_images(self, n=25):
        """
        Show misclassified images
        :return:
        """
        self.logger.info("Plotting and interpreting misclassified images. Please wait, I'm finalizing images.")
        fig = plt.figure(figsize=(50,100))

        images = self.misclassified[str(self.best_epoch)][:n]
        for i in range(1, n+1):
            j = 3*i - 2
            plt.subplot(n,3,j)
            plt.axis('off')
            imshow(images[i-1]["img"], self.config['std'], self.config['mean'], clip=True)
            plt.title("Pred : {} True : {}".format(self.id2classes[int(images[i-1]["pred"].cpu().numpy()[0])], 
                                                self.id2classes[int(images[i-1]["target"].cpu().numpy())]
                                            ))

            if self.config["interpret_image"]:
                heatmap, mask = self._interpret_images(images[i-1]["img"], self.id2classes[int(images[i-1]["target"].cpu().numpy())])
                plt.subplot(n, 3, j+1)
                plt.axis('off')
                plt.imshow(heatmap)
                plt.title("heatmap")
                plt.subplot(n, 3, j+2)
                plt.axis('off')
                plt.imshow(mask)
                plt.title("gradcam_mask")

        plt.tight_layout()
        fig.savefig(os.path.join(self.config["stats_dir"], 'misclassified_imgs.png'))
        self.logger.info("Misclassified Images saved.")

    def _interpret_images(self, image_data, label):
        """
        Interpret images using Grad Cam
        :return:
        """

        img = image_data.unsqueeze_(0).clone()
        # heatmaps = []
        # results = []
        # layers = ['layer1','layer2','layer3','layer4']
        layer = 'layer4'

        model_dict = dict(type='resnet', arch=self.model, layer_name=layer, input_size=(32, 32))
        gradcam = GradCam(model_dict)
        
        mask, _ = gradcam(img)
        heatmap, result = visualize_cam(mask, img)
        
        return np.clip(heatmap.permute(1,2,0).cpu().numpy(),0,1), np.clip(result.permute(1,2,0).cpu().numpy(),0,1)

        





