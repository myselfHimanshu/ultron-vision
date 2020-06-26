"""
Inference Agent for CIFAR10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torchsummary import summary

from agents.base import BaseAgent
from networks.threelayer_net import main as Net
from infdata.loader.tinyimagenet_dl import DataLoader as dl

# utils function
from utils.misc import *
from utils.gradcam.main import GradCam
from utils.gradcam.misc import visualize_cam

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


class TinyImageNetIAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("INFERENCE MODE ACTIVATED!!!")
        self.config = config
        self.use_cuda = self.config['use_cuda']
        self.visualize_inline = self.config['visualize_inline']

        # create network instance
        self.model = Net(self.config["num_classes"])

        # define data loader
        self.dataloader = dl(config=self.config)

        # intitalize classes

        self.classes_dict = self.dataloader.classes2idx
        self.classes = list({k: v for k, v in sorted(self.classes_dict.items(), key=lambda item: item[1])}.keys())
        self.id2classes = {y:i for i,y in self.classes_dict.items()}
        
        self.testclasses = self.dataloader.testclasses
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

    def show_per_class_accuracy(self):
        """
        Show per class acurracy for best saved weights
        :return:
        """
        self.model.to(self.device)
        self.model.eval()
        confusion_matrix = torch.zeros(len(self.classes), len(self.classes))
        with torch.no_grad():
            for i, (data, target) in enumerate(self.dataloader.valid_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                outputs = self.model(data)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(target.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
        class_acc = (100.*confusion_matrix.diag()/confusion_matrix.sum(1)).cpu().numpy()
        class_acc = zip(self.classes, class_acc)
        for class_name, acc_score in class_acc:
            self.logger.info(f"Accuracy of class : {class_name}\t\t{acc_score:4f}")

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


    def plot_graphs(self):
        """
        Plot accuracy and validation graph for train and valid dataset
        :return:
        """
        with open(self.stats_file_name) as f:
            data = json.load(f)

        train_acc = data["train_acc"]
        valid_acc = data["valid_acc"]
        train_loss = data["train_loss"]
        valid_loss = data["valid_loss"]

        epoch_count = range(1, self.config["epochs"]+1)
        fig = plt.figure(figsize=(5,5))
        
        plt.plot(epoch_count, train_acc)
        plt.plot(epoch_count, valid_acc)
        plt.plot(epoch_count, train_loss)
        plt.plot(epoch_count, valid_loss)
        plt.legend(["train_acc", "valid_acc", "train_loss", "valid_loss"])
        plt.xlabel('Epoch')
        plt.ylabel("Acc & Loss")
        if self.visualize_inline:
            plt.show();

        fig.savefig(os.path.join(self.config["stats_dir"], 'graph.png'))
        self.logger.info("Graphs saved.")

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
        fig = plt.figure(figsize=(5,5))
        
        plt.plot(epoch_count, train_acc)
        plt.plot(epoch_count, valid_acc)
        plt.legend(["train_acc","valid_acc"])
        plt.xlabel('Epoch')
        plt.ylabel("Accuracy")
        if self.visualize_inline:
            plt.show();

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
        fig = plt.figure(figsize=(5,5))
        
        plt.plot(epoch_count, train_loss)
        plt.plot(epoch_count, valid_loss)
        plt.legend(["train_loss","valid_loss"])
        plt.xlabel('Epoch')
        plt.ylabel("Loss")
        if self.visualize_inline:
            plt.show();

        fig.savefig(os.path.join(self.config["stats_dir"], 'loss.png'))
        self.logger.info("Loss Graph saved.")

    def plot_lr_graph(self):
        """
        Plot lr values graph
        :return:
        """
        with open(self.stats_file_name) as f:
            data = json.load(f)

        lr_values = data["lr_list"]
        
        epoch_count = range(1, self.config["epochs"]+1)
        fig = plt.figure(figsize=(5,5))
        
        plt.plot(epoch_count, lr_values)
        plt.xlabel('Epoch')
        plt.ylabel("LearningRate")
        if self.visualize_inline:
            plt.show();

        fig.savefig(os.path.join(self.config["stats_dir"], 'lr.png'))
        self.logger.info("Learning Rate Graph saved.")

    def show_misclassified_images(self, n=25):
        """
        Show misclassified images
        :return:
        """
        self.logger.info("Plotting and interpreting misclassified images. Please wait, I'm finalizing images.")
        fig = plt.figure(figsize=(10,20))

        images = self.misclassified[str(self.best_epoch)][:n]
        for i in range(1, n+1):
            j = 3*i - 2
            plt.subplot(n,3,j)
            plt.axis('off')
            imshow(images[i-1]["img"], self.config['std'], self.config['mean'], clip=True)
            plt.title("Pred : {} | True : {}".format(self.id2classes[int(images[i-1]["pred"].cpu().numpy()[0])], 
                                                self.id2classes[int(images[i-1]["target"].cpu().numpy())]
                                            ))

            if self.config["interpret_image"]:
                heatmap, mask = self._interpret_images(images[i-1]["img"], self.id2classes[int(images[i-1]["target"].cpu().numpy())], self.config['interpret_layer'])
                plt.subplot(n, 3, j+1)
                plt.axis('off')
                plt.imshow(heatmap)
                plt.title("heatmap")
                plt.subplot(n, 3, j+2)
                plt.axis('off')
                plt.imshow(mask)
                plt.title("gradcam_mask")

        plt.tight_layout()
        if self.visualize_inline:
            plt.show()
        fig.savefig(os.path.join(self.config["stats_dir"], 'misclassified_imgs.png'))
        self.logger.info("Misclassified Images saved.")

    def _interpret_images(self, image_data, label, layer='layer4'):
        """
        Interpret images using Grad Cam
        :return: heatmap and mask numpy array
        """
        self.model.to(torch.device('cpu'))
        img = image_data.unsqueeze_(0).clone()
        
        model_dict = dict(type='resnet', arch=self.model, layer_name=layer, input_size=(32, 32))
        gradcam = GradCam(model_dict)
        
        mask, _ = gradcam(img)
        heatmap, result = visualize_cam(mask, img)
        
        return np.clip(heatmap.permute(1,2,0).cpu().numpy(),0,1), np.clip(result.permute(1,2,0).cpu().numpy(),0,1)

        





