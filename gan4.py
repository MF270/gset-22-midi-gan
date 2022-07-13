import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, num_feats, hidden_units=256, drop_prob=0.6, use_cuda=False):
		super(Generator, self).__init__()


		# parameters
		self.hidden_dim = hidden_units
		self.use_cuda = use_cuda
		self.num_feats = num_feats

		self.fc_layer1 = nn.Linear(in_features=(num_feats*2), out_features=hidden_units)
		self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
		self.dropout = nn.Dropout(p=drop_prob)
		self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)