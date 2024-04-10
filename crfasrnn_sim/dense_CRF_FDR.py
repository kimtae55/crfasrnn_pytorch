import torch
import torch.nn as nn
from torch.distributions import Normal
from crfasrnn_sim.crfrnn import CrfRnn
import scipy.stats as stats
import numpy as np
from crfasrnn_sim.util import qvalue

class DenseCRFFDR(nn.Module):
	def __init__(self, im): 
		super(DenseCRFFDR, self).__init__()
		self.image = im
		self.crfrnn = CrfRnn(num_labels=2, num_iterations=5, image=im)
		self._softmax = torch.nn.Softmax(dim=1)
		self.p_value = 2.0*(1.0-stats.norm.cdf(np.fabs(im.numpy().copy())))
		self.q_sig = qvalue(self.p_value.ravel(), threshold=0.3)[0].reshape((1,1,) + im.shape[-3:])

		self.h = torch.rand((1,2,) + im.shape[-3:])
		self.h[:, 0:1, :, :, :] = torch.tensor(1.0-self.q_sig) # P(h=0|x)
		self.h[:, 1:2, :, :, :] = torch.tensor(self.q_sig)  # P(h=1|x)

		self.w_0 = nn.Parameter(torch.tensor([0.0])) # test this what's a good initial value
		self.N_01 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
		self.update_f1(self.h) 

	def forward(self, image):
		logits = self.h.clone()

		unary = self.h.clone()
		unary_rh = -1.0*(self.w_0 + self.N_01.log_prob(image) - self.f_1) # not adding w_0 here, try it out after logic fix
		unary *= unary_rh

		h_new = self.crfrnn(image, logits, unary)  # Assuming crfrnn returns a tuple with the first element being the updated h
		self.h = h_new.detach().clone()  # Detach h_new from the computation graph before updating self.h
		return h_new, self.f_1

	def update_f1(self, h): 
		h_flat = h[:,1].flatten() 
		image_flat = self.image.flatten() 
		indices = torch.where(h_flat > 0.5)[0] 

		if indices.nelement() == 0:  # Check if indices is empty
			print("No elements found greater than 0.5, using default f_1 update.")
			f1_flat = np.zeros(image_flat.shape)  # Or any other default logic
		else:	
			print('indices.nelement(): ', indices.nelement())
			h_flat_eq_1 = np.array(image_flat[indices])
			try:
				kde = stats.gaussian_kde(h_flat_eq_1)
				estimates = kde.logpdf(h_flat_eq_1)
				f1_flat = np.zeros(image_flat.shape)
				f1_flat[indices] = estimates
			except:
				f1_flat = np.zeros(image_flat.shape)  # Fallback to default in case of error

		f1 = f1_flat.reshape(self.image.shape)
		self.f_1 = torch.tensor(f1, dtype=torch.float32)
		print('update_f1() --> self.f_1: ', self.f_1)


class NllLoss(nn.Module):
	def __init__(self):
		super(NllLoss, self).__init__()
		self.N_01 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

	def forward(self, h, image, f_1):
		likelihood = (1-h)*self.N_01.log_prob(image).exp() + h*f_1.exp()
		'''
		print('h: ', h)
		print('f_1: ', f_1)
		print('likelihood: ', likelihood)
		'''
		loss = -torch.sum(torch.log(likelihood)) 

		print('NllLoss > loss: ', loss)
		return loss
