from __future__ import division
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from steerable.Spyr_PyTorch import Spyr_PyTorch

class STSIM_M(torch.nn.Module):
	def __init__(self, weights_path=None):
		super(STSIM_M, self).__init__()
		self.M = nn.Linear(82, 1, bias=False)

		if weights_path is not None:
			weights = torch.load(weights_path)
			with torch.no_grad():
				self.M.weight.copy_(weights['M.weight'])

	def init_weight(self, X):
		'''
		:param X: [dim of features, N]
		'''
		weights = 1/X.var(0).unsqueeze(0)
		with torch.no_grad():
			self.M.weight.copy_(weights)

	def forward_once(self, X):
		pass

	def forward(self, X1, X2, Y, mask):
		pred = torch.sqrt(self.M((X1 - X2)**2))
		coeff = self.criterion(pred, Y, mask)
		return coeff, pred

	def criterion(self, X, Y, mask):
		# Borda's rule of pearson coeff between X&Y
		coeff = 0
		N = mask.max().item()+1
		for i in range(N):
			X1 = X[mask==i,0].double()
			X1 = X1 - X1.mean()
			X2 = Y[mask==i].double()
			X2 = X2 - X2.mean()

			nom = torch.dot(X1, X2)
			denom = torch.sqrt(torch.sum(X1**2)*torch.sum(X2**2))

			coeff += torch.abs(nom/denom)
		return coeff/N

class Metric:
	# implementation of STSIM global (no sliding window), as the global version has a better performance, and also easier to implement
	def __init__(self, filter, device=None):
		self.device = torch.device('cpu') if device is None else device
		self.C = 1e-3
		self.filter = filter

	def STSIM(self, img1, img2, sub_sample=True):
		assert img1.shape == img2.shape
		assert len(img1.shape) == 4  # [N,C,H,W]
		assert img1.shape[1] == 1	# gray image

		s = Spyr_PyTorch(self.filter, sub_sample = sub_sample, device = self.device)

		pyrA = s.getlist(s.buildSpyr(img1))
		pyrB = s.getlist(s.buildSpyr(img2))

		stsim = map(self.pooling, pyrA, pyrB)

		return torch.mean(torch.stack(list(stsim)), dim=0).T # [BatchSize, FeatureSize]

	def STSIM2(self, img1, img2, sub_sample=True):
		assert img1.shape == img2.shape

		s = Spyr_PyTorch(self.filter, sub_sample = sub_sample, device = self.device)

		pyrA = s.buildSpyr(img1)
		pyrB = s.buildSpyr(img2)
		stsimg2 = list(map(self.pooling, s.getlist(pyrA), s.getlist(pyrB)))

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

	def STSIM_M(self, imgs, sub_sample = True):
		'''
		:param imgs: [N,C=1,H,W]
		:return:
		'''
		s =  Spyr_PyTorch(self.filter, sub_sample = sub_sample, device = self.device)
		coeffs = s.buildSpyr(imgs)

		f = []
		# single subband statistics
		for c in s.getlist(coeffs):
			mean = torch.mean(c, dim = [1,2,3])
			var = torch.var(c, dim = [1,2,3])
			f.append(mean)
			f.append(var)

			c = c - mean.reshape([-1,1,1,1])
			f.append(torch.mean(c[:, :, :-1, :] * c[:, :, 1:, :], dim=[1, 2, 3]) / var)
			f.append(torch.mean(c[:, :, :, :-1] * c[:, :, :, 1:], dim=[1, 2, 3]) / var)

		# correlation statistics
		# across orientations
		for orients in coeffs[1:-1]:
			for (c1, c2) in list(itertools.combinations(orients, 2)):
				c1 = torch.abs(c1)
				c1 = c1 - torch.mean(c1, dim = [1,2,3]).reshape([-1,1,1,1])
				c2 = torch.abs(c2)
				c2 = c2 - torch.mean(c2, dim = [1,2,3]).reshape([-1,1,1,1])
				denom = torch.sqrt(torch.var(c1, dim = [1,2,3]) * torch.var(c2, dim = [1,2,3]))
				f.append(torch.mean(c1*c2, dim = [1,2,3])/denom)

		for orient in range(len(coeffs[1])):
			for height in range(len(coeffs) - 3):
				c1 = torch.abs(coeffs[height + 1][orient])
				c1 = c1 - torch.mean(c1, dim=[1, 2, 3]).reshape([-1,1,1,1])
				c2 = torch.abs(coeffs[height + 2][orient])
				c2 = c2 - torch.mean(c2, dim=[1, 2, 3]).reshape([-1,1,1,1])
				c1 = F.interpolate(c1, size=c2.shape[2:])
				denom = torch.sqrt(torch.var(c1, dim = [1,2,3]) * torch.var(c2, dim = [1,2,3]))
				f.append(torch.mean(c1*c2, dim = [1,2,3])/denom)
		return torch.stack(f).T # [BatchSize, FeatureSize]

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
		mu1 = torch.mean(img1, dim = [1, 2, 3])
		mu2 = torch.mean(img2, dim = [1, 2, 3])

		sigma1_sq = torch.mean(img1**2, dim = [1,2,3]) - mu1 * mu1
		sigma1 = torch.sqrt(sigma1_sq)
		sigma2_sq = torch.mean(img2**2, dim = [1,2,3]) - mu2 * mu2
		sigma2 = torch.sqrt(sigma2_sq)

		Cmap = (2*sigma1*sigma2 + self.C)/(sigma1_sq + sigma2_sq + self.C)
		return Cmap

	def compute_C01_term(self, img1, img2):
		img11 = img1[..., :-1]
		img12 = img1[..., 1:]
		img21 = img2[..., :-1]
		img22 = img2[..., 1:]

		mu11 = torch.mean(img11, dim = [1,2,3])
		mu12 = torch.mean(img12, dim = [1,2,3])
		mu21 = torch.mean(img21, dim = [1,2,3])
		mu22 = torch.mean(img22, dim = [1,2,3])

		sigma11_sq = torch.mean(img11**2, dim = [1,2,3]) - mu11**2
		sigma12_sq = torch.mean(img12**2, dim = [1,2,3]) - mu12**2
		sigma21_sq = torch.mean(img21**2, dim = [1,2,3]) - mu21**2
		sigma22_sq = torch.mean(img22**2, dim = [1,2,3]) - mu22**2

		sigma1_cross = torch.mean(img11*img12, dim = [1,2,3]) - mu11*mu12
		sigma2_cross = torch.mean(img21*img22, dim = [1,2,3]) - mu21*mu22


		rho1 = (sigma1_cross + self.C) / (torch.sqrt(sigma11_sq * sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C) / (torch.sqrt(sigma21_sq * sigma22_sq) + self.C)

		C01map = 1 - 0.5*torch.abs(rho1 - rho2)

		return C01map

	def compute_C10_term(self, img1, img2):
		img11 = img1[:,:, :-1, :]
		img12 = img1[:,:, 1: , :]
		img21 = img2[:,:, :-1, :]
		img22 = img2[:,:, 1: , :]

		mu11 = torch.mean(img11, dim = [1,2,3])
		mu12 = torch.mean(img12, dim = [1,2,3])
		mu21 = torch.mean(img21, dim = [1,2,3])
		mu22 = torch.mean(img22, dim = [1,2,3])

		sigma11_sq = torch.mean(img11**2, dim = [1,2,3]) - mu11**2
		sigma12_sq = torch.mean(img12**2, dim = [1,2,3]) - mu12**2
		sigma21_sq = torch.mean(img21**2, dim = [1,2,3]) - mu21**2
		sigma22_sq = torch.mean(img22**2, dim = [1,2,3]) - mu22**2

		sigma1_cross = torch.mean(img11*img12, dim = [1,2,3]) - mu11*mu12
		sigma2_cross = torch.mean(img21*img22, dim = [1,2,3]) - mu21*mu22

		rho1 = (sigma1_cross + self.C)/(torch.sqrt(sigma11_sq)*torch.sqrt(sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C)/(torch.sqrt(sigma21_sq)*torch.sqrt(sigma22_sq) + self.C)

		C10map = 1 - 0.5*torch.abs(rho1 - rho2)

		return C10map

	def compute_cross_term(self, img11, img12, img21, img22):
		mu11 = torch.mean(img11, dim = [1,2,3])
		mu12 = torch.mean(img12, dim = [1,2,3])
		mu21 = torch.mean(img21, dim = [1,2,3])
		mu22 = torch.mean(img22, dim = [1,2,3])

		sigma11_sq = torch.mean(img11**2, dim = [1,2,3]) - mu11**2
		sigma12_sq = torch.mean(img12**2, dim = [1,2,3]) - mu12**2
		sigma21_sq = torch.mean(img21**2, dim = [1,2,3]) - mu21**2
		sigma22_sq = torch.mean(img22**2, dim = [1,2,3]) - mu22**2
		sigma1_cross = torch.mean(img11*img12, dim = [1,2,3]) - mu11*mu12
		sigma2_cross = torch.mean(img21*img22, dim = [1,2,3]) - mu21*mu22

		rho1 = (sigma1_cross + self.C)/(torch.sqrt(sigma11_sq*sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C)/(torch.sqrt(sigma21_sq*sigma22_sq) + self.C)

		Crossmap = 1 - 0.5*torch.abs(rho1 - rho2)
		return Crossmap
