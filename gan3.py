import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.fc1 = nn.Linear(1280, 600)
		self.fc2 = nn.Linear(600, 300)
		self.fc3 = nn.Linear(300, 100)
		self.fc4 = nn.Linear(100, 20)
		self.fc5 = nn.Linear(20, 1)

	def forward(self, x):
		x = nn.LeakyReLU(self.fc1(x), 0.2)
		x = nn.LeakyReLU(self.fc2(x), 0.2)
		x = nn.LeakyReLU(self.fc3(x), 0.2)
		x = nn.LeakyReLU(self.fc4(x), 0.2)
		x = nn.LeakyReLU(self.fc5(x), 0.2)
		x = nn.Sigmoid()
		return self.disc(x)

#midi dimension = 1281

class Generator(nn.Module):
	def __init__(self, z_dim, m_dim):
		super().__init__()
		self.gen = nn.Sequential(
			nn.Linear(z_dim, 256),
			nn.LeakyReLU(0.1),
			nn.Linear(256, m_dim), #1280, 256 x 5
			nn.Tanh(),
		)

	def forward(self, x):
		return self.gen(x)

#hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64 # 128, 256, or smaller
midi_dim = 256 * 5
batch_size = 32
num_epochs = 50

disc = Discriminator(midi_dim).to(device)
gen = Generator(z_dim, midi_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)



opt+disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummmaryWriter(f"runs/GAN_MIDI/fake")
writer_real = SummaryWriter(f"runs/GAN_MIDI/real")
step = 0

for epoch in range(num_epochs):
	for batch_idx, (real, _) in enumerate(loader):
		real = real.