"""
Main Agent for MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

from agents.base import BaseAgent
from networks.mnist_net import Net
from infdata.loader.mnist_dl import DataLoader as mnist_dl
from utils.misc import *

import os
curr_dir = os.path.dirname(__file__)

class MNISTAgent(BaseAgent):

    def __init__(self, config, use_cuda):
        super().__init__(config)
        self.config = config
        self.use_cuda = use_cuda

        # create network instance
        self.model = Net()

        # define data loader
        self.dataloader = mnist_dl(config=self.config, use_cuda=use_cuda)

        # define loss
        self.loss = nn.NLLLoss()

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

        # initialize loss and accuray arrays
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        # initialize misclassified data
        self.misclassified = {}

        # initialize maximum accuracy
        self.max_accuracy = 0.0

        # save checkpoint
        self.save_checkpoint = self.config['save_checkpoint']

        # set cuda flag
        self.use_cuda = use_cuda

        if self.use_cuda and self.config['cuda']:
            self.logger.info('WARNING : You have CUDA device, you should probably enable CUDA.')

        # set manual seed
        self.manual_seed = self.config['seed']
        if self.use_cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.model.to(self.device)
            
            self.logger.info("Program will RUN on ****GPU-CUDA****\n")
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device('cpu')
            self.logger.info("Program will RUN on ****CPU****\n")

        if self.config["load_checkpoint"]:
            self.load_checkpoint(self.config['checkpoint_file'])

    def load_checkpoint(self, file_name):
        """
        Latest Checkpoint loader
        :param file_name: name of checkpoint file
        :return:
        """
        file_name = os.path.join(curr_dir, "../checkpoints", file_name)
        checkpoint = torch.load(file_name)

        self.model = torch.load_state_dict(checkpoint['model'])
        self.optimizer = torch.load_state_dict(checkpoint['optimizer'])
        
        is_best = checkpoint["is_best"]
        if is_best:
            self.max_accuracy = checkpoint['test_accuracy']
        
        self.misclassified_data = checkpoint['misclassified_data']
        
    def save_checkpoint(self, file_name='checkpoint.pth.tar', is_best=1):
        """
        Checkpoint Saver
        :param file_name: name of checkpoint file path
        :param is_best: boolean flag indicating current metrix is best so far
        :return:   
        """
        checkpoint = {
            'epoch' : self.current_epoch,
            'test_accuracy' : self.max_accuracy,
            'misclassified_data' : self.misclassified,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'is_best' : is_best
        }

        file_name = os.path.join(curr_dir, "../checkpoints/mnist", file_name)
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
            for data, target in self.dataloader.test_loader:
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
                
            total_loss = running_loss/len(self.dataloader.test_loader.dataset)
            total_acc = 100. * running_correct/len(self.dataloader.test_loader.dataset)

        if(save_checkpoint and total_acc>self.max_accuracy):
            self.max_accuracy = total_acc
            try:
                self.save_checkpoint()
                self.logger.info("Saved Best Model")
            except Exception as e:
                self.logger.info(e)










        

    
