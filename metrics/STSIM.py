from __future__ import division
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from filterbank.Spyr_PyTorch import Spyr_PyTorch
from filterbank.SCFpyr_PyTorch import SCFpyr_PyTorch
from filterbank.DCT import DCT
from filterbank.sp3Filters import sp3Filters

class MyLinear(nn.Module):
	# parameters are between 0 and 1
	def __init__(self, input_size, output_size):
		super().__init__()
		self.W = nn.Parameter(torch.zeros(input_size, output_size))
		nn.init.xavier_uniform(self.W)

	def forward(self, x):
		return torch.mm(x, self.W**2)

class Metric:
	# implementation of STSIM global (no sliding window), as the global version has a better performance, and also easier to implement
	def __init__(self, filter=None, device=None, sub_sample=True):
		self.device = torch.device('cpu') if device is None else device
		self.C = 1e-10
		self.filter = filter
		if self.filter == 'SF':
			self.fb = Spyr_PyTorch(sp3Filters, sub_sample = sub_sample, device = self.device)
		elif self.filter == 'SCF':
			self.fb = SCFpyr_PyTorch(sub_sample = sub_sample, device = self.device)
		elif self.filter == 'DCT':
			self.fb = DCT(device = self.device)

	def STSIM1(self, img1, img2):
		assert img1.shape == img2.shape
		assert len(img1.shape) == 4  # [N,C,H,W]
		assert img1.shape[1] == 1	# gray image

		pyrA = self.fb.build(img1)
		pyrB = self.fb.build(img2)

		pyrA = self.fb.getlist(pyrA)	# magnitude, because pytorch version 1.6.0 doesn't support complex, and version >= 1.8.0 doesn't support 3080 right now
		pyrB = self.fb.getlist(pyrB)

		stsim = map(self.pooling, pyrA, pyrB)

		return torch.mean(torch.stack(list(stsim)), dim=0).T # [BatchSize, FeatureSize]

	def STSIM2(self, img1, img2):
		assert img1.shape == img2.shape

		pyrA = self.fb.build(img1)
		pyrB = self.fb.build(img2)
		stsimg2 = list(map(self.pooling, self.fb.getlist(pyrA), self.fb.getlist(pyrB))) # magnitude, because pytorch version 1.6.0 doesn't support complex, and version >= 1.8.0 doesn't support 3080 right now

		if self.filter == 'SCF':	# complex to real
			for i in range(1,4):
				for j in range(0,4):
					pyrA[i][j] = torch.view_as_complex(pyrA[i][j]).abs()
			for i in range(1,4):
				for j in range(0,4):
					pyrB[i][j] = torch.view_as_complex(pyrB[i][j]).abs()

		Nor = len(pyrA[1])

		# Accross scale, same orientation
		for scale in range(2, len(pyrA) - 1):
			for orient in range(Nor):
				img11 = pyrA[scale - 1][orient]
				img12 = pyrA[scale][orient]
				img11 = F.interpolate(img11, size=img12.shape[2:])

				img21 = pyrB[scale - 1][orient]
				img22 = pyrB[scale][orient]
				img21 = F.interpolate(img21, size=img22.shape[2:])

				stsimg2.append(self.compute_cross_term(img11, img12, img21, img22))

		# Accross orientation, same scale
		for scale in range(1, len(pyrA) - 1):
			for orient in range(Nor - 1):
				img11 = pyrA[scale][orient]
				img21 = pyrB[scale][orient]

				for orient2 in range(orient + 1, Nor):
					img13 = pyrA[scale][orient2]
					img23 = pyrB[scale][orient2]
					stsimg2.append(self.compute_cross_term(img11, img13, img21, img23))

		return torch.mean(torch.stack(stsimg2), dim=0).T # [BatchSize, FeatureSize]

	def STSIM(self, imgs):
		'''
		:param imgs: [N,C=1,H,W]
		:return: [N, feature dim] STSIM-features
		'''
		coeffs = self.fb.build(imgs)
		if self.filter == 'SCF':	# complex to real
			for i in range(1,4):
				for j in range(0,4):
					coeffs[i][j] = torch.sqrt(coeffs[i][j][..., 0]**2 + coeffs[i][j][..., 1]**2)

		f = []
		# single subband statistics
		for c in self.fb.getlist(coeffs):
			mean = torch.mean(c, dim = [1,2,3])
			var = torch.var(c, dim = [1,2,3])
			f.append(mean)
			f.append(var)

			c = c - mean.reshape([-1,1,1,1])
			f.append(torch.mean(c[:, :, :-1, :] * c[:, :, 1:, :], dim=[1, 2, 3]) / (var + self.C))
			f.append(torch.mean(c[:, :, :, :-1] * c[:, :, :, 1:], dim=[1, 2, 3]) / (var + self.C))

		# correlation statistics
		# across orientations
		for orients in coeffs[1:-1]:
			for (c1, c2) in list(itertools.combinations(orients, 2)):
				c1 = torch.abs(c1)
				c1 = c1 - torch.mean(c1, dim = [1,2,3]).reshape([-1,1,1,1])
				c2 = torch.abs(c2)
				c2 = c2 - torch.mean(c2, dim = [1,2,3]).reshape([-1,1,1,1])
				denom = torch.sqrt(torch.var(c1, dim = [1,2,3]) * torch.var(c2, dim = [1,2,3]))
				f.append(torch.mean(c1*c2, dim = [1,2,3])/(denom + self.C))

		for orient in range(len(coeffs[1])):
			for height in range(len(coeffs) - 3):
				c1 = torch.abs(coeffs[height + 1][orient])
				c1 = c1 - torch.mean(c1, dim=[1, 2, 3]).reshape([-1,1,1,1])
				c2 = torch.abs(coeffs[height + 2][orient])
				c2 = c2 - torch.mean(c2, dim=[1, 2, 3]).reshape([-1,1,1,1])
				c1 = F.interpolate(c1, size=c2.shape[2:])
				denom = torch.sqrt(torch.var(c1, dim = [1,2,3]) * torch.var(c2, dim = [1,2,3]))
				f.append(torch.mean(c1*c2, dim = [1,2,3])/(denom + self.C))

		return torch.stack(f).T # [BatchSize, FeatureSize]

	def STSIM_M(self, X1, X2=None, weight=None):
		if weight is not None:
			if len(X1.shape) == 4:
				# the input are raw images, extract STSIM-M features
				with torch.no_grad():
					X1 = self.STSIM(X1)  # [N, dim of feature]
					X2 = self.STSIM(X2)
			pred = (X1-X2)/weight	#[N, dim of feature]
			pred = torch.sqrt(torch.sum(pred**2,1))
			return pred
		else:
			if len(X1.shape) == 4:
				# the input are raw images, extract STSIM-M features
				with torch.no_grad():
					X1 = self.STSIM(X1)  # [N, dim of feature]
			weight = X1.std(0)	#[dim of feature]
			return weight

	def STSIM_I(self, X1, X2=None, mask=None, weight=None):
		if weight is not None:
			return self.STSIM_M(X1,X2,weight)
		else:
			if len(X1.shape) == 4:
				# the input are raw images, extract STSIM-M features
				with torch.no_grad():
					X1 = self.STSIM(X1)  # [N, dim of feature]
			var = torch.zeros(X1.shape[1], device=self.device)
			for i in set(mask.detach().cpu().numpy()):
				X1_i = X1[mask==i]
				X1_i = X1_i - X1_i.mean(0)	# substract intra-class mean [N, dim of feature]
				var += (X1_i**2).sum(0)		# square sum	for all intra-class sample [dim of feature]
			return torch.sqrt(var/X1.shape[0])

	def pooling(self, img1, img2):
		tmp = self.compute_L_term(img1, img2) * self.compute_C_term(img1, img2) * self.compute_C01_term(img1, img2) * self.compute_C10_term(img1, img2)
		return tmp**0.25

	def compute_L_term(self, img1, img2):
		# expectation over a small window
		mu1 = torch.mean(img1, dim = [1,2,3])
		mu2 = torch.mean(img2, dim = [1,2,3])

		Lmap = (2 * mu1 * mu2 + self.C)/( mu1 * mu1 + mu2 * mu2 + self.C)
		return Lmap

	def compute_C_term(self, img1, img2):
		mu1 = torch.mean(img1, dim = [1, 2, 3]).reshape(-1,1,1,1)
		mu2 = torch.mean(img2, dim = [1, 2, 3]).reshape(-1,1,1,1)

		sigma1_sq = torch.mean((img1 - mu1)**2, dim = [1,2,3])
		sigma1 = torch.sqrt(sigma1_sq)
		sigma2_sq = torch.mean((img2 - mu2)**2, dim = [1,2,3])
		sigma2 = torch.sqrt(sigma2_sq)

		Cmap = (2*sigma1*sigma2 + self.C)/(sigma1_sq + sigma2_sq + self.C)
		return Cmap

	def compute_C01_term(self, img1, img2):
		img11 = img1[..., :-1]
		img12 = img1[..., 1:]
		img21 = img2[..., :-1]
		img22 = img2[..., 1:]

		mu11 = torch.mean(img11, dim = [1,2,3]).reshape(-1,1,1,1)
		mu12 = torch.mean(img12, dim = [1,2,3]).reshape(-1,1,1,1)
		mu21 = torch.mean(img21, dim = [1,2,3]).reshape(-1,1,1,1)
		mu22 = torch.mean(img22, dim = [1,2,3]).reshape(-1,1,1,1)

		sigma11_sq = torch.mean((img11 - mu11)**2, dim = [1,2,3])
		sigma12_sq = torch.mean((img12 - mu12)**2, dim = [1,2,3])
		sigma21_sq = torch.mean((img21 - mu21)**2, dim = [1,2,3])
		sigma22_sq = torch.mean((img22 - mu22)**2, dim = [1,2,3])

		sigma1_cross = torch.mean((img11 - mu11)*(img12 - mu12), dim = [1,2,3])
		sigma2_cross = torch.mean((img21 - mu21)*(img22 - mu22), dim = [1,2,3])

		rho1 = (sigma1_cross + self.C) / (torch.sqrt(sigma11_sq * sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C) / (torch.sqrt(sigma21_sq * sigma22_sq) + self.C)

		C01map = 1 - 0.5*torch.abs(rho1 - rho2)

		return C01map

	def compute_C10_term(self, img1, img2):
		return self.compute_C01_term(img1.permute(0,1,3,2), img2.permute(0,1,3,2))

	def compute_cross_term(self, img11, img12, img21, img22):
		mu11 = torch.mean(img11, dim = [1,2,3]).reshape(-1,1,1,1)
		mu12 = torch.mean(img12, dim = [1,2,3]).reshape(-1,1,1,1)
		mu21 = torch.mean(img21, dim = [1,2,3]).reshape(-1,1,1,1)
		mu22 = torch.mean(img22, dim = [1,2,3]).reshape(-1,1,1,1)

		sigma11_sq = torch.mean((img11 - mu11)**2, dim = [1,2,3])
		sigma12_sq = torch.mean((img12 - mu12)**2, dim = [1,2,3])
		sigma21_sq = torch.mean((img21 - mu21)**2, dim = [1,2,3])
		sigma22_sq = torch.mean((img22 - mu22)**2, dim = [1,2,3])
		sigma1_cross = torch.mean((img11 - mu11)*(img12 - mu12), dim = [1,2,3])
		sigma2_cross = torch.mean((img21 - mu21)*(img22 - mu22), dim = [1,2,3])

		rho1 = (sigma1_cross + self.C)/(torch.sqrt(sigma11_sq*sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C)/(torch.sqrt(sigma21_sq*sigma22_sq) + self.C)

		Crossmap = 1 - 0.5*torch.abs(rho1 - rho2)
		return Crossmap

class STSIM_M(torch.nn.Module):
	def __init__(self, dim, mode=0, filter=None, device=None):
		'''
		Args:
			mode: regression, STSIM-M
			weights_path:
		'''
		super(STSIM_M, self).__init__()

		self.device = torch.device('cpu') if device is None else device
		self.mode = mode
		self.filter = filter
		if self.mode == 0:  	# factorization
			self.linear = nn.Linear(dim[0], dim[1])
		elif self.mode == 1:  	# 3-layer neural net
			self.hidden = nn.Linear(dim[0], dim[0])
			self.predict = nn.Linear(dim[0], 1)
		elif self.mode == 2:  	# regression
			self.linear = nn.Linear(dim[0], 1)
		elif self.mode == 3:	# diagonal Mahalanobis
			#self.linear = nn.Linear(dim[0], 1)
			self.linear = MyLinear(dim[0], 1)

	def forward(self, X1, X2):
		'''
		Args:
			X1:
			X2:
		Returns:
		'''
		if len(X1.shape) == 4:
			# the input are raw images, extract STSIM-M features
			m = Metric(self.filter, device=self.device)
			with torch.no_grad():
				X1 = m.STSIM(X1)
				X2 = m.STSIM(X2)
		if self.mode == 0:  # STSIM_Mf
			pred = self.linear(torch.abs(X1 - X2))  # [N, dim]
			pred = torch.bmm(pred.unsqueeze(1), pred.unsqueeze(-1)).squeeze(-1)  # inner-prod
			return torch.sqrt(pred)  # [N, 1]
		elif self.mode == 1:  # 3-layer neural net STSIM-NN
			pred = F.relu(self.hidden(torch.abs(X1 - X2)))
			pred = torch.sigmoid(self.predict(pred))
			return pred
		elif self.mode == 2:  # regression
			pred = self.linear(torch.abs(X1 - X2))  # [N, 1]
			return torch.sigmoid(pred)
		elif self.mode == 3:  # STSIM (diagonal) data driven STSIM-Md
			pred = self.linear(torch.abs(X1 - X2)**2)  # [N, 1]
			return torch.sqrt(pred)

if __name__ == '__main__':

	import pdb;pdb.set_trace()