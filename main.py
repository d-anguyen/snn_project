import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

#import tqdm
import models
import datasets
import plot
import train

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
batch_size = 256
dataset = 'dynamic'
dynamic_input = True
num_steps = 2
seed = np.random.randint(100) #tried values: 37,67,30
#seed = 30
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
path = './images/'


# Define Network
net = models.SNN(num_steps=num_steps,dynamic_input=dynamic_input).to(device)
print(net)


# Load data
if dataset == 'linear':
    toydata_train = datasets.Linear_ToyDataset(size=2048, seed=3)
    toydata_test = datasets.Linear_ToyDataset(size=400, seed=123)
elif dataset == 'relu':
    toydata_train = datasets.ReLU_ToyDataset(size=2048, seed=4)
    toydata_test = datasets.ReLU_ToyDataset(size=400, seed=123)
elif dataset == 'dynamic':
    toydata_train = datasets.Dynamic_ToyDataset(size=2048, seed=4)
    toydata_test = datasets.Dynamic_ToyDataset(size=400, seed = 123)
    
train_loader = DataLoader(toydata_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(toydata_test, batch_size=batch_size, shuffle=True)

# Visualize data and input partition before training
if dataset=='dynamic':
    plot.get_plot_dynamic(None, train_loader,label='target',dataset=dataset, path = path)
else:
    plot.get_plot(None,train_loader,label='target',dataset=dataset, path=path)
    plot.get_plot(net, train_loader, label='prediction',dataset=dataset, path=path)
plot.count_regions(net,num=100, box_size = 1, layer_indices = [0,1], path=path, train=False)

# Train
train_loss_hist, test_loss_hist = train.train(net, num_steps, train_loader, test_loader, num_epochs = 1000)
train.plot_learning_curve(train_loss_hist, test_loss_hist)

# Visualize data and input partition after training
if dataset=='dynamic':
    plot.get_plot_dynamic(None, train_loader,label='target',dataset=dataset, path = path)
else:
    plot.get_plot(None,train_loader,label='target',dataset=dataset, path=path)
    plot.get_plot(net, train_loader, label='prediction',dataset=dataset, path=path)
plot.count_regions(net,num=100, box_size = 1, layer_indices = [0,1], path=path, train=True)

