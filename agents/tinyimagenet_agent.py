"""
Main Agent for CIFAR10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR
import torch.nn.functional as F

from tqdm import tqdm

from agents.base import BaseAgent
from networks.resnet_net import ResNet18 as Net
from infdata.loader.tinyimagenet_dl import DataLoader as dl

# utils function
from utils.misc import *
from utils.lr_finder.lrfinder import LRFinder

from torchsummary import summary
import json
import os
import numpy as np


class TinyImageNetAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("TRAINING MODE ACTIVATED!!!")
        self.config = config
        self.use_cuda = self.config['use_cuda']
        self.visualize_inline = self.config['visualize_inline']

        # create network instance
        self.model = Net(self.config["num_classes"])

        # define data loader
        self.dataloader = dl(config=self.config)

        # intitalize classes
        self.classes_dict = self.dataloader.classes2idx
        self.id2classes = {y:i for i,y in self.classes_dict.items()}
        
        # define loss
        self.loss = nn.CrossEntropyLoss()

        #find optim lr and set optimizer
        self._find_optim_lr()

        # intialize weight decay
        self.l1_decay = self.config['l1_decay']
        self.l2_decay = self.config['l2_decay']

        # initialize step lr
        self.use_scheduler = self.config['use_scheduler']

        if self.use_scheduler:
            self.scheduler = self.config["scheduler"]["name"]
            if self.scheduler=="OneCycleLR":
                self.scheduler = OneCycleLR(self.optimizer, self.config['learning_rate'], 
                                            steps_per_epoch = len(self.dataloader.train_loader), 
                                            **self.config["scheduler"]["kwargs"]
                                            )
            else:
                self.logger.info("WARNING : OneCycleLr Scheduler was not setup. Re-initializing use_scheduler to False")
                self.use_scheduler = False
            
        # initialize Counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.best_epoch = 0

        # intitalize lr values list
        self.lr_list = []

        # initialize loss and accuray arrays
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []

        # initialize misclassified data
        self.misclassified = {}

        # initialize maximum accuracy
        self.max_accuracy = 0.0

        if not self.use_cuda and torch.cuda.is_available():
            self.logger.info('WARNING : You have CUDA device, you should probably enable CUDA.')

        # set manual seed
        self.manual_seed = self.config['seed']
        if self.use_cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config['gpu_device'])
            if torch.cuda.device_count() > 1:
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=(0,1))
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            
            self.logger.info("Program will RUN on ****GPU-CUDA****")
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device('cpu')
            self.logger.info("Program will RUN on ****CPU****")

        # summary of network
        print("****************************")
        print("**********NETWORK SUMMARY**********")
        summary(self.model, input_size=tuple(self.config['input_size']))
        print(self.model, file=open(os.path.join(self.config["summary_dir"],"model_arch.txt"), "w"))
        print("****************************")

        self.stats_file_name = os.path.join(self.config["stats_dir"], self.config["model_stats_file"])

    def load_checkpoint(self, file_name):
        """
        Latest Checkpoint loader
        :param file_name: name of checkpoint file
        :return:
        """
        file_name = os.path.join(self.config["checkpoint_dir"], file_name)
        checkpoint = torch.load(file_name, map_location='cpu')

        self.model = Net(self.config["num_classes"])
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=self.config['momentum'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        is_best = checkpoint["is_best"]
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

    def _find_optim_lr(self):
        """
        find optim learning rate to train network
        :return:
        """
        # self.logger.info("FINDING OPTIM LEARNING RATE...")
        # self.optimizer = optim.SGD(self.model.parameters(), lr=1e-7, momentum=self.config['momentum'])
        # lr_finder = LRFinder(self.model, self.optimizer, self.loss, device='cuda')
        # num_iter = (len(self.dataloader.train_loader.dataset)//self.config["batch_size"])*3
        # lr_finder.range_test(self.dataloader.train_loader, end_lr=1000, num_iter=num_iter)

        # if self.visualize_inline:
        #     lr_finder.plot()

        # history = lr_finder.history
        # optim_lr = history["lr"][np.argmin(history["loss"])] 
        # self.logger.info("Learning rate with minimum loss : " + str(optim_lr))
        # lr_finder.reset()
        
        # # set optimizer to optim learning rate
        # self.config["learning_rate"] = round(optim_lr,3)
        self.config["learning_rate"] = 0.01
        self.logger.info(f"Setting optimizer to optim learning rate : {self.config['learning_rate']}")
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], momentum=self.config['momentum'])

    def visualize_set(self):
        """
        Visualize Train set
        :return:
        """
        dataiter = iter(self.dataloader.train_loader)
        images, labels = dataiter.next()
        path = os.path.join(self.config["stats_dir"], 'training_images.png')

        visualize_data(images, self.config['std'], self.config['mean'], 30, self.visualize_inline, labels.cpu().numpy(), self.id2classes, path=path)


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
            for param_group in self.optimizer.param_groups:
                self.lr_list.append(param_group['lr'])
                self.logger.info(f"Current lr value = {param_group['lr']}")
            
            self.train_one_epoch()
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
            if self.use_scheduler:
                self.scheduler.step()

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
                    "valid_loss" : self.valid_losses, "valid_acc" : self.valid_acc,
                    "lr_list" : self.lr_list}
        
        with open(self.stats_file_name, "w") as f:
            json.dump(result, f)

        
        

        


