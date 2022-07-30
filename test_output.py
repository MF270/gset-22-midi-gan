from justdisc import Discriminator
from torch import Tensor
import torch
import csv
from pathlib import Path
num=50

results = []
for idx,file in enumerate(Path(r"C:\PythonPrograms\gset\midi\csv\musical").glob("**/*.csv")):
	if idx>1000:
		break
	with open(str(file),"r") as csv_file:
				messages = []
				reader = csv.reader(csv_file,delimiter=",")
				for line in reader:
					if len(line) == 1:
						continue
					for cell in line:
						messages.append(float(cell))
				t =Tensor(messages)

	with torch.no_grad():
		sd = torch.load(f"C:\PythonPrograms\gset\midi-gan\justdisc\disc{num}.pt")
		model = Discriminator(1280)
		model.load_state_dict(sd)
		model.eval()
		out = float(model(t))
		results.append(round(out))
print(f"On musical data:{results.count(1)/len(results)}")

results = []
for idx,file in enumerate(Path(r"C:\PythonPrograms\gset\midi\csv\nonmusical").glob("**/*.csv")):
	if idx>1000:
		break
	with open(str(file),"r") as csv_file:
				messages = []
				reader = csv.reader(csv_file,delimiter=",")
				for line in reader:
					if len(line) == 1:
						continue
					for cell in line:
						messages.append(float(cell))
				t =Tensor(messages)

	with torch.no_grad():
		sd = torch.load(f"C:\PythonPrograms\gset\midi-gan\justdisc\disc{num}.pt")
		model = Discriminator(1280)
		model.load_state_dict(sd)
		model.eval()
		out = float(model(t))
		results.append(round(out))
print(f"On nonmusical data:{results.count(0)/len(results)}")
