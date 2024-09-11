import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import random
import time 
from crfasrnn_gibbs.dense_CRF_FDR import DenseCRFFDR, NllLoss
from crfasrnn_gibbs.util import p_lis, compute_kl_divergence
from crfasrnn_gibbs.util import qvalue
from crfasrnn_gibbs.crfsampler import CRFSampler
from crfasrnn_gibbs.crf_permuto_sampler import CRFPermutoSampler
#from torchviz import make_dot

def print_gradients(model):
	for name, param in model.named_parameters():
		if param.grad is not None:
			grad_min = param.grad.min().item()
			grad_max = param.grad.max().item()
			print(f"Gradient for {name}: min={grad_min}, max={grad_max}")
	else:
		print(f"Gradient for {name}: None")

def h_dist(h, savepath):
	# Flatten the array to get a 1D array of all values
	h_flat = h.flatten()

	# Plotting the histogram of the values
	plt.figure(figsize=(10, 6))
	plt.hist(h_flat, bins=100, edgecolor='k', alpha=0.7)
	plt.title('Distribution of Values in the 30x30x30 Array')
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.grid(True)

	# Saving the plot as a PNG file
	plt.savefig(os.path.join(savepath, 'distribution_of_values.png'))	

def main(args):
	if torch.cuda.is_available():
		print("CUDA is available. PyTorch is using GPU.")
	else:
		print("CUDA is not available. PyTorch is using CPU.")

	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)

	labelname_noext = '.'.join(args.labelpath.split('/')[-1].split('.')[:-1])

	method_dict = {
		'deepfdr': 0,
		# put other methods here if desired
	}

	metric = {
		'fdr': [[] for _ in range(len(method_dict))],
		'fnr': [[] for _ in range(len(method_dict))],
		'atp': [[] for _ in range(len(method_dict))]
	}
	def add_result(fdr, fnr, atp, method_index):
		metric['fdr'][method_index].append(fdr)
		metric['fnr'][method_index].append(fnr)
		metric['atp'][method_index].append(atp)

	times = []
	for i in range(args.replications):
		start = time.time()
		print('-----------------------------------------------')
		print('-------------------------------- RUN_NUMBER: ',i)
		print('-----------------------------------------------')
		model_name = 'lr_' + str(args.lr) + '.pth' # change the name 
		r = _train(args, i)
		print(f'epoch result: {r[0]},{r[1]},{r[2]}')
		add_result(r[0], r[1], r[2], 0) # deepfdr
		end = time.time()
		times.append(end-start)
	print('DL computation time: ', np.mean(times), np.std(times))

	for key, val in metric.items():
		print(key)
		for i in range(len(val)):
			print(f"{list(method_dict.keys())[i]} -- {val[i]}")

	# Save final signal_file
	with open(os.path.join(args.savepath, 'out_' + labelname_noext + '_' + os.path.splitext(model_name)[0]) + '.txt', 'w') as outfile:
		outfile.write('DL:\n')
		mfdr, sfdr = np.mean(metric['fdr'][0]), np.std(metric['fdr'][0])
		mfnr, sfnr = np.mean(metric['fnr'][0]), np.std(metric['fnr'][0])
		matp, satp = np.mean(metric['atp'][0]), np.std(metric['atp'][0])
		outfile.write(f'fdr: {mfdr} ({sfdr})\n')
		outfile.write(f'fnr: {mfnr} ({sfnr})\n')
		outfile.write(f'atp: {matp} ({satp})\n')
		# Write the metric dictionary to the file
		for key, val in metric.items():
			outfile.write(f'{key}:\n')
			for i in range(len(val)):
				outfile.write(f"{list(method_dict.keys())[i]} -- {val[i]}\n")

def _train(args, data_index):
	'''
	random.seed(1000)
	np.random.seed(1000)
	torch.manual_seed(1000)
	torch.cuda.manual_seed_all(1000)
	'''

	if args.mode == 'sim':
		X = np.load(args.datapath)[data_index:data_index+1].reshape((1,1,30,30,30))
		y = np.load(args.labelpath).reshape((1,1,30,30,30))

		X = torch.FloatTensor(X) 

		net = DenseCRFFDR(X)
		optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
		crit = NllLoss()

		losses = []
		kl_ds = []

		net.train()
		for epoch in range(args.e):
			optimizer.zero_grad()
			h, f_1, unary = net(X)
			#dot = make_dot(h, params=dict(net.named_parameters()))
			#dot.render(os.path.join(args.savepath, str(epoch)), format='png')

			f_1 = net.update_f1_(h) # try using the 'correct' version for loss function 
			loss, loss_hard=crit(h[:,1], X, f_1, unary, torch.FloatTensor(y)) # crucial!!! 
			loss.backward()

			torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
			print_gradients(net)

			optimizer.step()
			losses.append(loss.item())
			kl_ds.append(compute_kl_divergence(y,h[:,1]))
			net.update_f1(h)
			scheduler.step()

		losses_x = np.arange(1, len(losses)+1, 1)

		loss_savepath = os.path.join(args.savepath, str(args.lr) + '_' + str(args.e) + '_')

		plt.figure(figsize=(10, 6))  
		plt.plot(losses_x, losses, label='Training Loss', marker='o')  
		plt.title('Training Loss Over Epochs')  
		plt.xlabel('Epoch')  
		plt.ylabel('Loss')  
		plt.legend()  
		plt.grid(True)  
		plt.tight_layout()  
		plt.savefig(loss_savepath + 'losses.png')

		# save model if want
		#torch.save(net.state_dict(), loss_savepath + 'model.pth')

		# Assuming X is your input data and y is your ground truth labels
		net.eval()
		h_meanfield, f_1, unary = net(X)
		hard_labels = (h_meanfield[:,1] > 0.5).detach().numpy().squeeze()
		h_meanfield = h_meanfield.detach().numpy()

		print('meanfield: ', p_lis(h_meanfield[:,1].squeeze(), threshold=0.1, label=y.ravel(), savepath=args.savepath))

		# Perform Gibbs sampling
		# Initialize CRFSampler with the trained weights
		theta_alpha = net.crfrnn.params.alpha
		theta_beta = net.crfrnn.params.beta
		theta_gamma = net.crfrnn.params.gamma
		w1 = [net.crfrnn.spatial_ker_weights[0,0], net.crfrnn.spatial_ker_weights[1,1]]
		w2 = [net.crfrnn.bilateral_ker_weights[0,0], net.crfrnn.bilateral_ker_weights[1,1]]

		start = time.time()
		sampler = CRFPermutoSampler(burn_in = 0, 
							 num_samples = 10, 
							 trained_weights=[net.crfrnn.spatial_ker_weights, net.crfrnn.bilateral_ker_weights, net.crfrnn.compatibility_matrix],
							 image=X)

		h = sampler.gibbs_sampler_3d_parallel(hard_labels, X, unary).detach().numpy()  # P(h=1|x)
		end = time.time()
		print('time taken: ', end - start)
		h_dist(h.squeeze(), args.savepath)
		#visualize_3d_mesh(h.squeeze())

		print('meanfield: ', p_lis(h_meanfield[:,1].squeeze(), threshold=0.1, label=y.ravel(), savepath=args.savepath))
		fdr, fnr, atp = p_lis(h.squeeze(), threshold=0.1, label=y.ravel(), savepath=args.savepath)
		return fdr, fnr, atp

	elif args.mode == 'adni':
		print('boo')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
	parser.add_argument('--lr', default=2e-3, type=float)
	parser.add_argument('--e', default=20, type=int)
	parser.add_argument('--replications', default=1, type=int)
	parser.add_argument('--datapath', type=str)
	parser.add_argument('--labelpath', default='./', type=str)
	parser.add_argument('--savepath', default='./', type=str)
	parser.add_argument('--mode', default='sim', type=str) # sim, adni
	args = parser.parse_args()
	main(args)