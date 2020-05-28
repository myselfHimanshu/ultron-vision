"""
Main Agent for CIFAR10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from tqdm import tqdm

from agents.base import BaseAgent
from networks.cifar10_atrous_net import Cifar10AtrousNet as Net
# from networks.resnet_net import ResNet18 as Net
from infdata.loader.cifar10_dl import DataLoader as dl
from utils.misc import *

from torchsummary import summary
import json
import matplotlib.pyplot as plt

import os
curr_dir = os.path.dirname(__file__)

class Cifar10Agent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.use_cuda = self.config['use_cuda']

        # create network instance
        self.model = Net()

        # define data loader
        self.dataloader = dl(config=self.config)

        # intitalize classes
        self.classes = self.dataloader.classes
        self.id2classes = {i:y for i,y in enumerate(self.classes)}

        # define loss
        self.loss = nn.CrossEntropyLoss()

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=self.config['momentum'])

        # intialize weight decay
        self.l1_decay = self.config['l1_decay']
        self.l2_decay = self.config['l2_decay']

        # initialize step lr
        self.use_step_lr = self.config['use_step_lr']

        if self.use_step_lr:
            self.step_size = self.config['step_size']
            self.step_gamma = self.config['step_gamma']
            self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.step_gamma)

        # initialize Counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.best_epoch = 0

        # initialize loss and accuray arrays
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []

        # initialize misclassified data
        self.misclassified = {}

        # initialize maximum accuracy
        self.max_accuracy = 0.0

        # save checkpoint
        self.save_checkpoint = self.config['save_checkpoint']

        if not self.use_cuda and torch.cuda.is_available():
            self.logger.info('WARNING : You have CUDA device, you should probably enable CUDA.')

        # set manual seed
        self.manual_seed = self.config['seed']
        if self.use_cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config['gpu_device'])
            self.model = self.model.to(self.device)
            self.loss = self.model.to(self.device)
            
            self.logger.info("Program will RUN on ****GPU-CUDA****\n")
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device('cpu')
            self.logger.info("Program will RUN on ****CPU****\n")

        # summary of network
        print("****************************")
        print("**********NETWORK SUMMARY**********")
        summary(self.model, input_size=tuple(self.config['input_size']))
        print("****************************")


        if self.config["load_checkpoint"]:
            self.load_checkpoint(self.config['checkpoint_file'])

        self.stats_file_name = os.path.join(self.config["stats_dir"], self.config["model_stats_file"])


    def load_checkpoint(self, file_name):
        """
        Latest Checkpoint loader
        :param file_name: name of checkpoint file
        :return:
        """
        file_name = os.path.join(self.config["checkpoint_dir"], file_name)
        checkpoint = torch.load(file_name)

        self.model = torch.load_state_dict(checkpoint['model'])
        self.optimizer = torch.load_state_dict(checkpoint['optimizer'])
        
        is_best = checkpoint["is_best"]
        if is_best:
            self.max_accuracy = checkpoint['valid_accuracy']
        
        self.misclassified = checkpoint['misclassified_data']
        
    def save_checkpoint(self, file_name='checkpoint.pth.tar', is_best=1):
        """
        Checkpoint Saver
        :param file_name: name of checkpoint file path
        :param is_best: boolean flag indicating current metrix is best so far
        :return:   
        """
        checkpoint = {
            'epoch' : self.current_epoch,
            'valid_accuracy' : self.max_accuracy,
            'misclassified_data' : self.misclassified,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'is_best' : is_best
        }

        file_name = os.path.join(self.config["checkpoint_dir"], file_name)
        torch.save(checkpoint, file_name)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except Exception as e:
            self.logger.info(e)

    def train(self):
        """
        Main training iteration
        :return:
        """
        for epoch in range(1, self.config['epochs']+1):
            self.train_one_epoch()
            if self.use_step_lr:
                self.scheduler.step()
            self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()

        running_loss = 0.0
        running_correct = 0

        pbar = tqdm(self.dataloader.train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            
            if self.l1_decay>0.0:
                loss += regularize_loss(self.model, loss, self.l1_decay, 1)
            if self.l2_decay>0.0:
                loss += regularize_loss(self.model, loss, self.l2_decay, 2)
            
            loss.backward()
            self.optimizer.step()

            _, preds = torch.max(output.data, 1)

            # calculate running loss and accuracy
            running_loss += loss.item()
            running_correct += (preds==target).sum().item()
            pbar.set_description(desc=f'loss = {loss.item()} batch_id = {batch_idx}')

        total_loss = running_loss/len(self.dataloader.train_loader.dataset)
        total_acc = 100. * running_correct/len(self.dataloader.train_loader.dataset)

        self.train_losses.append(total_loss)
        self.train_acc.append(total_acc)
        self.logger.info(f"TRAIN EPOCH : {self.current_epoch}\tLOSS : {total_loss:.4f}\tACC : {total_acc:.4f}")

    def validate(self):
        """
        One cycle of model evaluation
        :return:
        """
        self.model.eval()

        running_loss = 0.0
        running_correct = 0

        with torch.no_grad():
            for data, target in self.dataloader.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                running_loss += self.loss(output, target).sum().item()
                pred = output.argmax(dim=1, keepdim=True)
                running_correct += pred.eq(target.view_as(pred)).sum().item()

                is_correct = pred.eq(target.view_as(pred))
                misclass_idx = (is_correct==0).nonzero()[:,0]
                for idx in misclass_idx:
                    if str(self.current_epoch) not in self.misclassified:
                        self.misclassified[str(self.current_epoch)] = []
                    self.misclassified[str(self.current_epoch)].append({
                        "target" : target[idx],
                        "pred" : pred[idx],
                        "img" : data[idx]
                    })
                
            total_loss = running_loss/len(self.dataloader.valid_loader.dataset)
            total_acc = 100. * running_correct/len(self.dataloader.valid_loader.dataset)

        if(self.config['save_checkpoint'] and total_acc>self.max_accuracy):
            self.max_accuracy = total_acc
            self.best_epoch = self.current_epoch
            try:
                self.save_checkpoint()
                self.logger.info("Saved Best Model")
            except Exception as e:
                self.logger.info(e)

        self.valid_losses.append(total_loss)
        self.valid_acc.append(total_acc)
        self.logger.info(f"VALID EPOCH : {self.current_epoch}\tLOSS : {total_loss:.4f}\tACC : {total_acc:.4f}")

    def finalize(self):
        """
        Finalize operations
        :return:
        """
        self.logger.info("Please wait while finalizing the operations.. Thank you")
        
        result = {"train_loss" : self.train_losses, "train_acc" : self.train_acc,
                    "valid_loss" : self.valid_losses, "valid_acc" : self.valid_acc}
        
        with open(self.stats_file_name, "w") as f:
            json.dump(result, f)

    
    def predict(self):
        try:
            if(self.dataloader.test_loader==None):
                self.logger.info("Test Loader is NOT DEFINED!!!")

            self.load_checkpoint(self.config['checkpoint_file'])
            self.model.eval()
            predictions = []

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    predictions.extend(pred.cpu().numpy())
            
            return predictions
        except Exception as e:
            self.logger.info("Test loader prediction FAILED!!!")
            self.logger.info(e)
            return []

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

    def show_misclassified_images(self, n=25):
        """
        Show misclassified images
        :return:
        """
        fig = plt.figure(figsize=(10,10))

        images = self.misclassified[str(self.best_epoch)][:n]
        for i in range(1, n+1):
            plt.subplot(5,5,i)
            plt.axis('off')
            plt.imshow(images[i-1]["img"].cpu().numpy()[0], cmap='gray_r')
            plt.title("Predicted : {} \nActual : {}".format(self.id2classes[images[i-1]["pred"][0].cpu().numpy()], 
                                                self.id2classes[images[i-1]["target"].cpu().numpy()]))

        plt.tight_layout()
        fig.savefig(os.path.join(self.config["stats_dir"], 'misclassified_imgs.png'))
