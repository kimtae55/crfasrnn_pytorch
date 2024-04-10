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
from crfasrnn_sim.dense_CRF_FDR import DenseCRFFDR, NllLoss
from crfasrnn_sim.util import p_lis, visualize_3d_mesh, compute_kl_divergence
from torchviz import make_dot

def main(args):
	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)

	if args.mode == 'sim':
		data_index = 0
		X = np.load(args.datapath)[data_index:data_index+1].reshape((1,1,30,30,30))
		y = np.load(args.labelpath)[data_index:data_index+1].reshape((1,1,30,30,30))

		X = torch.FloatTensor(X) 

		net = DenseCRFFDR(X)
		optimizer = torch.optim.AdamW(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
		crit = NllLoss()

		losses = []
		kl_ds = []

		net.train()
		for epoch in range(args.e):
			optimizer.zero_grad()
			h, f_1 = net(X)
			#dot = make_dot(h, params=dict(net.named_parameters()))
			#dot.render(os.path.join(args.savepath, str(epoch)), format='png')

			loss=crit(h[:,1], X, f_1)
			loss.backward()

			# Check for NaN gradients in all parameters
			for name, param in net.named_parameters():
				if param.grad is not None and torch.isnan(param.grad).any():
					raise ValueError(f'NaN gradient found in parameter: {name}')

			optimizer.step()
			losses.append(loss.item())
			kl_ds.append(compute_kl_divergence(y,h[:,1]))
			net.update_f1(h)

		losses_x = np.arange(1, len(losses)+1, 1)

		loss_savepath = os.path.join(args.savepath, str(args.lr) + '_' + str(args.e) + '_')

		plt.figure(figsize=(10, 6))  
		plt.plot(losses_x, losses, label='Training Loss', marker='o')  
		plt.plot(losses_x, kl_ds, label='KL Divergence', marker='x', linestyle='--')
		plt.title('Training Loss Over Epochs')  
		plt.xlabel('Epoch')  
		plt.ylabel('Loss')  
		plt.legend()  
		plt.grid(True)  
		plt.tight_layout()  
		plt.savefig(loss_savepath + 'losses.png')

		torch.save(net.state_dict(), loss_savepath + 'model.pth')

		net.eval()
		h = net(X)[0].detach().numpy() # P(h=1|x)
		print(h[:,1])
		fdr, fnr, atp = p_lis(h[:,1].squeeze(), threshold=0.1, label=y.ravel(), savepath=args.savepath)
		print(fdr, fnr, atp)
		visualize_3d_mesh(h[:,1].squeeze())

	elif args.mode == 'adni':
		print('boo')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--e', default=20, type=int)
	parser.add_argument('--datapath', type=str)
	parser.add_argument('--labelpath', default='./', type=str)
	parser.add_argument('--savepath', default='./', type=str)
	parser.add_argument('--mode', default='sim', type=str) # sim, adni
	args = parser.parse_args()
	main(args)