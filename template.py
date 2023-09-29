import warnings
warnings.filterwarnings('ignore')
import torch
import math
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ttach
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

class template:
    def __init__(self, class_num: int, train_dataset: Dataset, valid_dataset: Dataset, 
                 lr = 5e-4, batch_size = 16, num_workers = 4, drop_last = True, seed = 'seed', set_seed = True,
                 model = 'HarDNet68', loss_function = 'Cross Entropy', optimizer = 'AdamW',
                 pretrained = True, save_best = True, use_ttach = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if set_seed:
            self.set_seed(str(seed))
        self.device = device
        self.save_best = save_best
        self.class_num = class_num
        self.model = self.set_model(model, pretrained)
        if use_ttach:
            self.use_ttach = use_ttach
            self.ttach_model = ttach.ClassificationTTAWrapper(self.model, ttach.aliases.d4_transform(), merge_mode='mean')
        self.optimizer = self.set_optimizer(optimizer, lr)
        self.loss_function = self.set_loss_function(loss_function)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers,  drop_last = drop_last)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers,  drop_last = drop_last)

    def set_seed(self, seed):
        seed = math.prod([ord(i) for i in seed])%(2**32)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        pd.core.common.random_state(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def set_model(self, model_name, pretrained):
        if model_name == 'ResNet18':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.class_num)
            return model.to(self.device)
        if model_name == 'ResNet50':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained = pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.class_num)
            return model.to(self.device)
        if model_name == 'ResNet152':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained = pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.class_num)
            return model.to(self.device)
        if model_name == 'HarDNet68':
            if self.device == torch.device('cpu') or not pretrained:
                if pretrained:
                    print('HarDNet68 wit cpu cannot pretrained, using not pretrained model')
                model = torch.hub.load('PingoLH/Pytorch-HarDNet','hardnet68', pretrained = False, map_location = self.device).to(self.device)
            else:
                model = torch.hub.load('PingoLH/Pytorch-HarDNet','hardnet68', pretrained = True, map_location = self.device).to(self.device)
            model.base[-1] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Dropout(0.1),
                nn.Linear(1024, self.class_num)
            )
            return model.to(self.device)
        if model_name == 'HarDNet85':
            if self.device == torch.device('cpu') or not pretrained:
                if pretrained:
                    print('HarDNet85 wit cpu cannot pretrained, using not pretrained model')
                model = torch.hub.load('PingoLH/Pytorch-HarDNet','hardnet85', pretrained = False, map_location = self.device).to(self.device)
            else:
                model = torch.hub.load('PingoLH/Pytorch-HarDNet','hardnet85', pretrained = True, map_location = self.device).to(self.device)
            model.base[-1] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(1280, self.class_num)
            )
            return model.to(self.device)
        else:
            raise ValueError('Model ' + model_name + ' not found. Use get_model_list() to see the acceptable model.')
    
    def get_model(self):
        return self.model
        
    def get_model_list(self):
        model_list = ['ResNet18', 'ResNet50', 'ResNet152', 'HarDNet68', 'HarDNet85']
        print('Acceptable models: ', model_list)

    def set_optimizer(self, optimizer_name, lr):
        optimizer_list = {
            'SGD': torch.optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9, nesterov = True),
            'Adam': torch.optim.Adam(self.model.parameters(), lr = lr),
            'AdamW': torch.optim.AdamW(self.model.parameters(), lr = lr)
        }
        if optimizer_name in optimizer_list:
            return optimizer_list[optimizer_name]
        else:
            raise ValueError('Optimizer ' + optimizer_name + ' not found. Use get_optimizer_list() to see the acceptable optimizer.')
        
    def get_optimizer(self):
        return self.optimizer
    
    def get_optimizer_list(self):
        optimizer_list = ['AdamW']
        print('Acceptable optimizers: ', optimizer_list)
    
    def set_loss_function(self, loss_function_name):
        loss_function_list = {
            'Cross Entropy': torch.nn.CrossEntropyLoss()
        }
        if loss_function_name in loss_function_list:
            return loss_function_list[loss_function_name]
        else:
            raise ValueError('Loss function' + loss_function_name + ' not found. Use loss_function_list() to see the acceptable loss functions.')
        
    def get_loss_function(self):
        return self.loss_function
    
    def get_loss_function_list(self):
        loss_function_list = ['Cross Entropy']
        print('Acceptable loss functions: ', loss_function_list)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        with tqdm(self.train_dataloader, unit = 'Batch', desc = 'Train') as tqdm_loader:
            for index, (image, label) in enumerate(tqdm_loader):
                image = image.to(device = self.device)
                label = torch.tensor(label.to(device = self.device), dtype = torch.long)
                predict = self.model(image).to(device = self.device)
                loss = self.loss_function(predict, label)
                predict = predict.cpu().detach().argmax(dim = 1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.detach().item()
                total_loss = total_loss + loss
                accuracy = accuracy_score(predict, label.cpu())
                total_accuracy = total_accuracy + accuracy
            
                tqdm_loader.set_postfix(loss = loss, average_loss = total_loss/(index + 1), average_accuracy = total_accuracy/(index + 1))
    
    def valid_epoch(self, best_accuracy, best_loss):
        model = self.model
        if self.use_ttach:
            model = self.ttach_model
        model.eval()
        total_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            with tqdm(self.valid_dataloader, unit = 'Batch', desc = 'Valid') as tqdm_loader:
                for index, (image, label) in enumerate(tqdm_loader):
                    image = image.to(device = self.device)
                    label = torch.tensor(label.to(device = self.device), dtype = torch.long)
                    predict = model(image).to(device = self.device)
                    loss = self.loss_function(predict, label)
                    predict = predict.cpu().detach().argmax(dim = 1)

                    loss = loss.detach().item()
                    total_loss = total_loss + loss
                    accuracy = accuracy_score(predict, label.cpu())
                    total_accuracy = total_accuracy + accuracy
            
                    tqdm_loader.set_postfix(loss = loss, average_loss = total_loss/(index + 1), average_accuracy = total_accuracy/(index + 1))
                average_loss = total_loss/len(tqdm_loader)
        average_accuracy = total_accuracy/len(tqdm_loader)
        
        if average_accuracy > best_accuracy and self.save_best:
            print('Best model update.')
            best_accuracy = average_accuracy
            best_loss = average_loss
            self.best_model = self.model
            torch.save(self.model,'model.pkl')
        elif average_accuracy == best_accuracy and self.save_best:
            if average_loss <= best_loss:
                print('Best model update.')
                best_loss = average_loss
                self.best_model = self.model
                torch.save(self.model,'model.pkl')
        return best_accuracy, best_loss
    
    def train_and_valid(self, epoch = 40):
        best_accuracy = -100
        best_loss = 100
        for epoch in range(epoch):
            print('\nEpoch {}'.format(epoch + 1))
            self.train_epoch()
            best_accuracy, best_loss = self.valid_epoch(best_accuracy, best_loss)

        if self.save_best:
            print('Best accuracy: {:.4f}, Best loss: {:.4f}'.format(best_accuracy, best_accuracy))
            return self.best_model
        else:
            return self.model

if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    validset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
    model = template(model = 'HarDNet68', loss_function = 'Cross Entropy', optimizer = 'AdamW', train_dataset = trainset, valid_dataset = validset, class_num = 10, use_ttach = True)
    model.train_and_valid()