import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
	def __init__(self, in_features):
	