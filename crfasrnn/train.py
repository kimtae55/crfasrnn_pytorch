import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import random
import time 
from crfasrnn.dense_CRF_FDR import DenseCRFFDR, NllLoss
from crfasrnn.util import p_lis

torch.autograd.set_detect_anomaly(True)

def main(args):
	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)

	if args.mode == 'sim':
		data_index = 0
		X = np.load(args.datapath)[data_index:data_index+1].reshape((1,1,30,30,30))
		y = np.load(args.labelpath)[data_index:data_index+1].reshape((1,1,30,30,30))

		X = X[:, :, :10, :10, :10]
		y = y[:, :, :10, :10, :10]

		X = torch.FloatTensor(X) 

		net = DenseCRFFDR(X)
		optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
		crit = NllLoss()

		losses = []

		net.train()
		for epoch in range(args.e):
			optimizer.zero_grad()
			h, f_1 = net(X)
			loss=crit(h[:,1], X, f_1)
			loss.backward()
			print('done backpropagation')
			optimizer.step()
			losses.append(loss.item())

		losses_x = np.arange(1, len(losses)+1, 1)

		custom_savepath = os.path.join(args.savepath, str(args.lr) + '_' + str(args.e) + '_')

		plt.figure(figsize=(10, 6))  
		plt.plot(losses_x, losses, label='Training Loss', marker='o')  
		plt.title('Training Loss Over Epochs')  
		plt.xlabel('Epoch')  
		plt.ylabel('Loss')  
		plt.legend()  
		plt.grid(True)  
		plt.tight_layout()  
		plt.savefig(custom_savepath + 'losses.png')

		torch.save(net.state_dict(), custom_savepath + 'model.pth')

		net.eval()
		h = net(X) # P(h=1|x)

		fdr, fnr, atp = p_lis(gamma_1, threshold=0.1, label=y, savepath=args.savepath)
		print(fdr, fnr, atp)

	elif args.mode == 'adni':
		print('boo')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--e', default=10, type=int)
	parser.add_argument('--datapath', type=str)
	parser.add_argument('--labelpath', default='./', type=str)
	parser.add_argument('--savepath', default='./', type=str)
	parser.add_argument('--mode', default='sim', type=str) # sim, adni
	args = parser.parse_args()
	main(args)