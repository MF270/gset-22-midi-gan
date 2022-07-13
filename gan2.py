import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np

# Hyper-parameters
#latent_size is input nodes, hidden_size is size of hidden layers 
latent_size = 640
hidden_size = 256
num_epochs = 300
batch_size = 32
sample_dir = 'samples'
save_dir = 'save'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#mnist = torchvision.datasets.MNIST(root='./data/',
 #                                  train=True,
  #                                 transform=transform,
   #                                download=True)
#with open("filename.txt","r") as midiSet:

# Data loader
#data_loader = torch.utils.data.DataLoader(dataset=midiSet,
 #                                         batch_size=batch_size, 
  #                                        shuffle=True)

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, latent_size),
    nn.Tanh())

print(G)



# Device setting
#D = D.cuda()
G = G.cuda()

# Binary cross entropy loss and optimizer
#criterion = nn.BCELoss()
#d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
  out = (x + 1) / 2
  return out.clamp(0, 1)

def reset_grad():
 #   d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Statistics to be saved
d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)