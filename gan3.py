import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.disc = nn.Sequential(
			nn.Linear(int_features, 128),
			nn.LeakyReLU(0.1),
			nn.Linear(128, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		retirn self.disc(x)

#midi dimension = 1280

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

disc = Discrmininator(midi_dim).to(device)
gen = Generator(z_dim, midi_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms =


opt+disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummmaryWriter(f"runs/GAN_MIDI/fake")
writer_real = SummaryWriter(f"runs/GAN_MIDI/real")
step = 0

for epoch in range(num_epochs):
	for batch_idx, (real, _) in enumerate(loader):
		real = real.