import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import csv
from pathlib import Path
import math


DIR_TO_CSVS = r"C:\Users\sofia\Downloads\csv"
class Discriminator(nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.fc1 = nn.Linear(in_features, 600)
		self.fc2 = nn.Linear(600, 300)
		self.fc3 = nn.Linear(300, 100)
		self.fc4 = nn.Linear(100, 50)
		self.fc5 = nn.Linear(50, 20)
		self.fc6 = nn.Linear(20, 1)

	def forward(self, x):
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc2(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc3(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc4(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc5(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc6(x))
		return (x)

class Generator(nn.Module):
	def __init__(self, z_dim, m_dim):
		super().__init__()
		self.fc1 = nn.Linear(z_dim,100)
		self.fc2 = nn.Linear(100, 150)
		self.fc3 = nn.Linear(150, 200)
		self.fc4 = nn.Linear(200, 250)
		self.fc5 = nn.Linear(250, 300)
		self.fc6 = nn.Linear(300, 400)
		self.fc7 = nn.Linear(400, 600)
		self.fc8 = nn.Linear(600,800)
		self.fc9 = nn.Linear(800,1000)
		self.fc10 = nn.Linear(1000, m_dim)
		
	def forward(self, x):
		#no dropout layers because they could make it a little weaker
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		x = self.fc4(x)
		x = self.fc5(x)
		x = self.fc6(x)
		x = self.fc7(x)
		x = self.fc8(x)
		x = self.fc9(x)
		x = self.fc10(x)
		return(midify(x))

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
				if len(line) == 1:
					continue
				for cell in line:
					messages.append(int(cell))
		return Tensor(messages),label

def wsig(max, scale, x):
	y = max/(1+((math.e)**(-1*scale*x)))
	return y

def midify(x):
	L = x.tolist()
	#print(L)
	output = []
	for i in L:
		working = []
		for idx, element in enumerate(i):
			column = idx%5
			#note on/note off
			if column==0:
				val = round(wsig(1, 1, element))
			#channel
			elif column == 1:
				val = round(wsig(15, 1, element))
			#pitch, velocity
			elif column in (2, 3):
				val = round(wsig(127, 1, element))
			#time
			elif column == 4:
				val = round(wsig(480, 1, element))
			working.append(val)
		output.append(working)
	#print(output)
	return torch.Tensor(output)

def D_train(x):
	D.zero_grad()	
	x_real, y_real = x.view(-1,midi_dim), torch.ones(batch_size,1)
	x_real, y_real = Tensor(x_real.to(device)), Tensor(y_real.to(device))
	D_real = D(x_real)
	D_real_loss = criterion(D_real,y_real)

	z = Tensor(torch.randn(batch_size, z_dim).to(device))
	x_fake, y_fake = G(z), Tensor(torch.zeros(batch_size, 1).to(device))

	D_fake = D(x_fake)
	D_fake_loss = criterion(D_fake,y_fake)
	D_loss = D_real_loss + D_fake_loss
	D_loss.backward()
	D_optimizer.step()
	return D_loss.data.item()

def save_as_csv(t):
	with open(r"C:\Users\sofia\OneDrive\Documents\Python Code\output.csv","w",newline="") as csv_file:
		writer = csv.writer(csv_file)
		rest_of_list = t[0]
		out_list = [rest_of_list[i:i+5] for i in range(0, len(rest_of_list), 5)]
		for l in out_list:
			if len(l) != 5:
				continue
			x = l.squeeze().tolist()
			writer.writerow(x)

def G_train(epochs):
	G.zero_grad()
	z = Tensor(torch.randn(batch_size, z_dim).to(device))
	y = Tensor(torch.ones(batch_size, 1).to(device))
	G_output = G(z)
	D_output = D(G_output)
	G_loss = criterion(D_output, y)
	# gradient backprop & optimize ONLY G's parameters
	G_loss.backward()
	G_optimizer.step()
	if epochs%5 == 0:
		save_as_csv(G_output)
	return G_loss.data.item()

def pretrain_d(real,fake,epochs):
	for epoch in range(1,epochs+1):
		real_data, _ = next(iter(real))
		fake_data, _ = next(iter(fake))
		print(f"pre epoch {epoch}")
		D_train(real_data)
		D_train(fake_data)


if __name__=="__main__":
	#hyperparameters
	device = "cuda" if torch.cuda.is_available() else "cpu"
	lr = 5e-4
	z_dim = 100
	midi_dim = 256 * 5
	batch_size = 64
	num_epochs = 5

	D = Discriminator(midi_dim).to(device)
	G = Generator(z_dim, midi_dim).to(device)
	fixed_noise = torch.randn((batch_size, z_dim)).to(device)

	real_data = MidiDataset(fr"{DIR_TO_CSVS}\musical")
	fake_data = MidiDataset(fr"{DIR_TO_CSVS}\nonmusical")
	real_loader = DataLoader(real_data,batch_size=batch_size,shuffle=True,drop_last=True)
	fake_loader = DataLoader(fake_data, batch_size=batch_size, shuffle=True,drop_last=True)

	opt_disc = optim.Adam(D.parameters(), lr=lr)
	opt_gen = optim.Adam(G.parameters(), lr=lr)
	criterion = nn.MSELoss()
	#Generator uses MSELoss, 
	writer_fake = SummaryWriter(f"runs/GAN_MIDI/fake")
	writer_real = SummaryWriter(f"runs/GAN_MIDI/real")

	G_optimizer = optim.Adam(G.parameters(), lr = lr)
	D_optimizer = optim.Adam(D.parameters(), lr = lr)

	disc_extra_epochs = 1

	pretrain_d(real_loader,fake_loader,disc_extra_epochs)
	print("training full net")
	for epoch in range(1, num_epochs+1):

		G_losses, D_losses = [], []
		for  (x, _) in (real_loader):
			for i in range(5):
				G_losses.append(G_train(num_epochs))
			D_losses.append(D_train(x))
		d_loss_mean = torch.mean(Tensor(D_losses))
		g_loss_mean = torch.mean(Tensor(G_losses))
		print(f"epoch {epoch}/{num_epochs} loss_d: {(d_loss_mean)} loss_g: {(g_loss_mean)} ")
		if epoch%5 == 0:
			torch.save(G.state_dict(), rf"C:\Users\sofia\Downloads\csv{epoch}.pt")
			torch.save(D.state_dict(), rf"C:\Users\sofia\Downloads\csv{epoch}.pt")

		#freeze the disc if it's too good
		if d_loss_mean < 0.7* g_loss_mean:
			D.fc1.weight.requires_grad = False
			D.fc1.bias.requires_grad = False
			D.fc2.weight.requires_grad = False
			D.fc2.bias.requires_grad = False
			D.fc3.weight.requires_grad = False
			D.fc3.bias.requires_grad = False
			D.fc4.weight.requires_grad = False
			D.fc4.bias.requires_grad = False
			D.fc5.weight.requires_grad = False
			D.fc5.bias.requires_grad = False

		else:
			D.fc1.weight.requires_grad = True
			D.fc1.bias.requires_grad = True
			D.fc2.weight.requires_grad = True
			D.fc2.bias.requires_grad = True
			D.fc3.weight.requires_grad = True
			D.fc3.bias.requires_grad = True
			D.fc4.weight.requires_grad = True
			D.fc4.bias.requires_grad = True
			D.fc5.weight.requires_grad = True
			D.fc5.bias.requires_grad = True