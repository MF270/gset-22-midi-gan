import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import csv
from pathlib import Path
from math import log

DIR_TO_CSVS = r"C:\PythonPrograms\gset\midi\csv"
class Discriminator(nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.fc1 = nn.Linear(in_features, 600)
		self.fc2 = nn.Linear(600, 300)
		self.fc3 = nn.Linear(300, 100)
		self.fc4 = nn.Linear(100, 20)
		self.fc5 = nn.Linear(20, 1)

	def forward(self, x):
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc2(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc3(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc4(x))
		x = F.dropout(x, 0.3)
		return torch.sigmoid(self.fc5(x))

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
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc7(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc8(x))
		x = F.dropout(x, 0.3)
		x = F.leaky_relu(self.fc9(x))
		x = F.dropout(x, 0.3)

		#not sure if should use sigmoid, can use tanh or softmax
		return (self.fc10(x))
		# return torch.sigmoid(self.fc10(x))
		# x = torch.sigmoid(self.fc5(x))
		# return x

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
		return Tensor(messages),label

# def D_train(x):
# 	D.zero_grad()	
# 	x_real, y_real = x.view(-1,midi_dim), torch.ones(batch_size,1)
# 	x_real, y_real = Tensor(x_real.to(device)), Tensor(y_real.to(device))
# 	# try:
# 	D_real = D(x_real)
# 	#what does this do?

# 	z = Tensor(torch.randn(batch_size, z_dim).to(device))
# 	x_fake, y_fake = G(z), Tensor(torch.zeros(batch_size, 1).to(device))
# 	#this seems to be creating fake data from the generator, which really should be nonmusical fake data, right?

# 	D_fake = D(x_fake)

# 	D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
# 	D_loss.backward()
# 	D_optimizer.step()
# 	return D_loss.data.item()
def inv_sig(x):
	if x==0:
		x=1e-4
	if x == 1:
		return 64
	try:
		return -log((1-x)/x)
	except:
		print(x)
def save_as_csv(t):
	with open("fuckinghelpme.csv","w",newline="") as csv_file:
		# with torch.no_grad():
		# 	t.detach().apply_(inv_sig)
		writer = csv.writer(csv_file)
		writer.writerow([t[0][0].tolist()])
		rest_of_list = t[0][1:]
		out_list = [rest_of_list[i:i+5] for i in range(0, len(rest_of_list))]
		for l in out_list:
			if len(l) != 5:
				continue
			x = l.squeeze().tolist()
			writer.writerow(x)
		# out_list = [int((i.squeeze())) for i in out_list]
		# writer.writerows(out_list)

# def G_train(n):
# 	G.zero_grad()
# 	z = Tensor(torch.randn(batch_size, z_dim).to(device))
# 	y = Tensor(torch.ones(batch_size, 1).to(device))
# 	G_output = G(z)
# 	D_output = D(G_output)
# 	# D_fake = D(x_fake)
# 	G_loss = -torch.mean(D_fake)
# 	if (n%2 == 0):
# 		save_as_csv(G_output)
# 	G_loss.backward()
# 	G_optimizer.step()
# 	return G_loss.data.item()

# def pretrain_d(real,fake,epochs):
# 	for epoch in range(1,epochs+1):
# 		real_data, _ = next(iter(real))
# 		fake_data, _ = next(iter(fake))
# 		print(f"pre epoch {epoch}")
# 		D_train(real_data)
# 		D_train(fake_data)

def train(x,n):
	D.zero_grad()
	G.zero_grad()
	x_real, y_real = x.view(-1,midi_dim), torch.ones(batch_size,1)
	x_real, y_real = Tensor(x_real.to(device)), Tensor(y_real.to(device))
	D_real = D(x_real)

	z = Tensor(torch.randn(batch_size, z_dim).to(device)).detach()
	x_fake, y_fake = G(z), Tensor(torch.zeros(batch_size, 1).to(device))
	#this seems to be creating fake data from the generator, which really should be nonmusical fake data, right?

	D_fake = D(x_fake)

	D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
	D_loss.backward()
	D_optimizer.step()

	z = Tensor(torch.randn(batch_size, z_dim).to(device))
	y = Tensor(torch.ones(batch_size, 1).to(device))
	D_output = D(x_fake)
	G_loss = -torch.mean(D_fake)
	if (n%2 == 0):
		save_as_csv(x_fake)

	G_loss.backward(retain_graph=True)
	G_optimizer.step()
	return D_loss.data.item(), G_loss.data.item()


if __name__=="__main__":
	#hyperparameters
	device = "cuda" if torch.cuda.is_available() else "cpu"
	lr = 5e-4
	z_dim = 100 # 128, 256, or smaller
	midi_dim = 256 * 5 + 1
	batch_size = 64
	num_epochs = 100

	D = Discriminator(midi_dim).to(device)
	G = Generator(z_dim, midi_dim).to(device)
	fixed_noise = torch.randn((batch_size, z_dim)).to(device)

	real_data = MidiDataset(fr"{DIR_TO_CSVS}\musical")
	fake_data = MidiDataset(fr"{DIR_TO_CSVS}\nonmusical")
	real_loader = DataLoader(real_data,batch_size=batch_size,shuffle=True,drop_last=True)
	fake_loader = DataLoader(fake_data, batch_size=batch_size, shuffle=True,drop_last=True)

	opt_disc = optim.Adam(D.parameters(), lr=lr)
	opt_gen = optim.Adam(G.parameters(), lr=lr)
	criterion = nn.BCELoss()
	gencriterion = nn.MSELoss()
	writer_fake = SummaryWriter(f"runs/GAN_MIDI/fake")
	writer_real = SummaryWriter(f"runs/GAN_MIDI/real")

	G_optimizer = optim.Adam(G.parameters(), lr = lr)
	D_optimizer = optim.Adam(D.parameters(), lr = lr)

	disc_extra_epochs = 0

	# pretrain_d(real_loader,fake_loader,disc_extra_epochs)
	print("training full net")
	for epoch in range(1, num_epochs+1):
		print(f"epoch {epoch}")
		if epoch%2 == 0:
			torch.save(G.state_dict(), rf"C:\PythonPrograms\gset\midi-gan\models\gen{epoch}.pt")
			torch.save(D.state_dict(), rf"C:\PythonPrograms\gset\midi-gan\models\disc{epoch}.pt")

		G_losses, D_losses = [], []
		for (x, _) in (real_loader):
			dloss, gloss = train(x,epoch)
			D_losses.append(dloss)
			G_losses.append(gloss)
		# print(f'[{epoch}/{num_epochs}]: loss_d: {torch.mean(Tensor(D_losses))}, loss_g: {torch.mean(Tensor(G_losses))}')