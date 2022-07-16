from gan3 import Generator,Discriminator
from torch import Tensor
import torch
import csv
with open(r"C:\PythonPrograms\gset\midi\csv\musical\Ain't Nuthin' but a G Thang\19.csv","r") as csv_file:
			messages = []
			reader = csv.reader(csv_file,delimiter=",")
			for line in reader:
				for cell in line:
					messages.append(float(cell))
			t =Tensor(messages)

with torch.no_grad():
	sd = torch.load("C:\PythonPrograms\gset\midi-gan\models\disc100.pt")
	model = Discriminator(1281)
	model.load_state_dict(sd)
	model.eval()
	print(int(model(t)))