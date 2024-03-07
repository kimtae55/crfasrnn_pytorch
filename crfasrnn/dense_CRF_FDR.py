import torch
import torch.nn as nn
from torch.distributions import Normal
from crfasrnn.crfrnn import CrfRnn
from scipy.stats import gaussian_kde
import numpy as np

class DenseCRFFDR(nn.Module):
	def __init__(self, im):
		super(DenseCRFFDR, self).__init__()
		self.image = im
		self.crfrnn = CrfRnn(num_labels=2, num_iterations=10, image=im)
		self._softmax = torch.nn.Softmax(dim=0)
		self._epsilon = 1e-8
		self.h = self._softmax(torch.rand((1,2,) + im.shape[-3:]))
		self.w_0 = nn.Parameter(torch.rand(im.shape)) 
		self.N_01 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
		self.update_f1() 

	def forward(self, image):
		unary = self.h*(self.w_0 + self.N_01.log_prob(image) - torch.log(self.f_1 + self._epsilon))# 
		unary = unary.squeeze(1)

		self.h = self.crfrnn(image, unary) # mean field inference, so output should be between [0,1]. Check this!
		print(self.h.shape)
		return self.h, self.f_1

	def update_f1(self):
		h_flat = self.h[:,1].flatten()
		image_flat = self.image.flatten()
		indices = torch.where(h_flat > 0.5)[0]
		h_flat_eq_1 = np.array(image_flat[indices])
		kde = gaussian_kde(h_flat_eq_1)
		estimates = kde(h_flat_eq_1)
		f1_flat = np.zeros(image_flat.shape)
		f1_flat[indices] = estimates
		f1 = f1_flat.reshape(self.image.shape)
		self.f_1 = torch.tensor(f1, dtype=torch.float32)

class NllLoss(nn.Module):
	def __init__(self):
		super(NllLoss, self).__init__()
		self.N_01 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

	def forward(self, h, image, f_1):
		likelihood = (1-h)*self.N_01.log_prob(image).exp() + h*f_1
		'''
		print('h: ', h)
		print('f_1: ', f_1)
		print('likelihood: ', likelihood)
		'''
		loss = -torch.sum(torch.log(likelihood))

		print('loss: ', loss)
		return loss