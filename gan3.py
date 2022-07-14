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
midi_dim = 256 * 5 + 1
batch_size = 32
num_epochs = 100

D = Discriminator(midi_dim).to(device)
G = Generator(z_dim, midi_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

real_data = MidiDataset(fr"{DIR_TO_CSVS}\musical")
fake_data = MidiDataset(fr"{DIR_TO_CSVS}\nonmusical")
real_loader = DataLoader(real_data,batch_size=batch_size,shuffle=True)
fake_loader = DataLoader(fake_data, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(D.parameters(), lr=lr)
opt_gen = optim.Adam(G.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MIDI/fake")
writer_real = SummaryWriter(f"runs/GAN_MIDI/real")

G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


def D_train(x):
	D.zero_grad()	
	x_real, y_real = x.view(-1,midi_dim), torch.ones(batch_size,1)
	x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

	D_output = D(x_real)
	D_real_loss = criterion(D_output, y_real) #y_real should just be all 1s
	D_real_score = D_output
	#what does this do?

	z = Variable(torch.randn(batch_size, z_dim).to(device))
	x_fake, y_fake = G(z), Variable(torch.zeros(batch_size, 1).to(device))
	#this seems to be creating fake data from the generator, which really should be nonmusical fake data, right?

	D_output = D(x_fake)
	#what is the function being called on D? also ew globals
	D_fake_loss = criterion(D_output, y_fake) #y_fake should just be all 0s
	D_fake_score = D_output
	#what is the fake score
	D_loss = D_real_loss + D_fake_loss
	D_loss.backward()
	D_optimizer.step()
	return D_loss.data.item()

def G_train(x):
    G.zero_grad()
    z = Variable(torch.randn(batch_size, z_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)
    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()

def pretrain_d(real,fake,epochs):
	for epoch in range(1,epochs+1):
		real_data, _ = next(iter(real))
		fake_data, _ = next(iter(fake))
		D.train(real_data)
		D.train(fake_data)

disc_extra_epochs = 20

pretrain_d(real_loader,fake_loader)		
for epoch in range(1, num_epochs+1):
	G_losses, D_losses = [], []
	for (x, _) in (real_loader):
		D_losses.append(D_train(x))
		G_losses.append(G_train(x))
	print(f'[{epoch}/{num_epochs}]: loss_d: {torch.mean(torch.FloatTensor(D_losses))}, loss_g: {torch.mean(torch.FloatTensor(G_losses))}')
