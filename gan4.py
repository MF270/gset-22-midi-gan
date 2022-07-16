from gan3 import Generator,Discriminator
from torch import Tensor
import torch
import csv
with open(r"C:\PythonPrograms\fuckinghelpme.csv","r") as csv_file:
			messages = []
			reader = csv.reader(csv_file,delimiter=",")
			for line in reader:
				for cell in line:
					messages.append(float(cell))
			t =Tensor(messages)

with torch.no_grad():
	sd = torch.load("C:\PythonPrograms\gset\midi-gan\models\disc40.pt")
	model = Discriminator(1281)
	model.load_state_dict(sd)
	model.eval()
	print(model.state_dict())
	# print(t.size())
	# print(float(model(t)))
with torch.no_grad():
	sd = torch.load("C:\PythonPrograms\gset\midi-gan\models\disc60.pt")
	model2 = Discriminator(1281)
	model2.load_state_dict(sd)
	model2.eval()
	print(model.state_dict())
	# print(t.size())
	# print(float(model(t)))

print(model2.state_dict()-model.state_dict())