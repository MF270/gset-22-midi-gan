import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import csv
from pathlib import Path


DIR_TO_CSVS = r"C:\PythonPrograms\gset\midi\csv"
class Discriminator(nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.fc1 = nn.Linear(1281, 600)
		self.fc2 = nn.Linear(600, 300)
		self.fc3 = nn.Linear(300, 100)
		self.fc4 = nn.Linear(100, 20)
		self.fc5 = nn.Linear(20, 1)

	def forward(self, x):
		x = F.LeakyReLU(self.fc1(x), 0.2)
		x = F.dropout(x, 0.3)
		x = F.LeakyReLU(self.fc2(x), 0.2)
		x = F.dropout(x, 0.3)
		x = F.LeakyReLU(self.fc3(x), 0.2)
		x = F.dropout(x, 0.3)
		x = F.LeakyReLU(self.fc4(x), 0.2)
		x = F.dropout(x, 0.3)
		return nn.Sigmoid(self.fc5(x))

#midi dimension = 1281

class Generator(nn.Module):
	def __init__(self, z_dim, m_dim):
		super().__init__()
		self.fc1 = nn.Linear(1,20)
		self.fc2 = nn.Linear(20, 100)
		self.fc3 = nn.Linear(100, 300)
		self.fc4 = nn.Linear(300, 600)
		self.fc5 = nn.Linear(600, 1281)
		
	def forward(self, x):
		x = F.LeakyReLU(self.fc1(x), 0.2)
		x = F.dropout(x, 0.3)
		x = F.LeakyReLU(self.fc2(x), 0.2)
		x = F.dropout(x, 0.3)
		x = F.LeakyReLU(self.fc3(x), 0.2)
		x = F.dropout(x, 0.3)
		x = F.LeakyReLU(self.fc4(x), 0.2)
		x = F.dropout(x, 0.3)
		#not sure if should use sigmoid, can use tanh or softmax
		return nn.sigmoid(self.fc5(x))

class MidiDataset(Dataset):
	def __init__(self,dir):
		self.dir = dir
		self.files = list(Path(self.dir).glob("**/*.csv"))

	def __len__(self):
		return len((self.files))

	def __getitem__(self,index):
		file = self.files[index]
		container = str(file.parent).split("\\")[-1]
		label = 1 if container == "musical" else 0
		with open(str(file),"r") as csv_file:
			messages = []
			reader = csv.reader(csv_file,delimiter=",")
			for line in reader:
				for cell in line:
					messages.append(int(cell))
		return messages,label

#hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64 # 128, 256, or smaller
midi_dim = 256 * 5
batch_size = 32
num_epochs = 50

D = Discriminator(midi_dim).to(device)
gen = Generator(z_dim, midi_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

midi_data = MidiDataset(DIR_TO_CSVS)
training_loader = DataLoader(midi_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(midi_data, batch_size=64, shuffle=True)

opt_disc = optim.Adam(D.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MIDI/fake")
writer_real = SummaryWriter(f"runs/GAN_MIDI/real")


class MidiDataset(Dataset):
	def __init__(self,dir):
		self.dir = dir
		self.files = list(Path(self.dir).glob("**/*.csv"))

	def __len__(self):
		return len((self.files))

	def __getitem__(self,index):
		file = self.files[index]
		container = str(file.parent).split("\\")[-1]
		label = 1 if container == "musical" else 0
		with open(str(file),"r") as csv_file:
			messages = []
			reader = csv.reader(csv_file,delimiter=",")
			for line in reader:
				for cell in line:
					messages.append(int(cell))
		return messages,label

def D_train(x):
	D.zero_grad()

	#get the real data to do the yeah 

	D_output = D(x_real)
	D_real_loss = criterion(D_output, y_real) #y_real should just be all 1s
	D_real_score = D_output

	#get the fake data to do the yeah

	D_output = D(x_fake)
	D_fake_loss = criterion(D_output, y_fake) #y_fake should just be all 0s
	D_fake_score = D_output

	D_loss = D_real_loss + D_fake_loss
	D_loss.backward()
	D_optimizer.step()

	return D_Loss.data.item()

def G_train(x):
	G.zero_grad()

	#idk how u get the stuff

	G_output = G(z)
	D_output = D(G_output)
	G_loss = criterion(D_output, y)

	G_loss.backward()
	G_optimizer.step()

	return G_loss.data.item()

