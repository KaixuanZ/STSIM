from __future__ import division
import numpy as np
import itertools

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from steerable.Spyr_PyTorch import Spyr_PyTorch

def fspecial(win, sigma, device):
	"""
	2D gaussian mask - should give the same result as MATLAB's
	fspecial('gaussian',[shape],[sigma])
	"""
	shape = (win, win)
	m, n = [(ss-1.)/2. for ss in shape]
	y, x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()

	if sumh != 0:
		h /= sumh
	return torch.from_numpy(h).to(device).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


class Metric:
	def __init__(self, filter, win=7, device=None):
		self.device = torch.device('cpu') if device is None else device
		self.win = win
		self.k = fspecial(win, win/6, self.device)	#[out_C, in_C/group, H, W]
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

		return torch.mean(torch.stack(list(stsim)), dim=0)

	def STSIM2(self, img1, img2):
		assert img1.shape == img2.shape

		s = Spyr_PyTorch(sub_sample = True, device = self.device)
		s_nosub = Spyr_PyTorch(sub_sample = False, device = self.device)

		pyrA = s.getlist(s.buildSpyr(img1))
		pyrB = s.getlist(s.buildSpyr(img2))
		stsimg2 = list(map(self.pooling, pyrA, pyrB))

		# Add cross terms
		bandsAn = s_nosub.buildSpyr(img1)
		bandsBn = s_nosub.buildSpyr(img2)

		Nor = len(bandsAn[1])

		# Accross scale, same orientation
		for scale in range(2, len(bandsAn) - 1):
			for orient in range(Nor):
				img11 = self.abs(bandsAn[scale - 1][orient])
				img12 = self.abs(bandsAn[scale][orient])

				img21 = self.abs(bandsBn[scale - 1][orient])
				img22 = self.abs(bandsBn[scale][orient])

				stsimg2.append(self.compute_cross_term(img11, img12, img21, img22).mean(dim=[1,2,3]))

		# Accross orientation, same scale
		for scale in range(1, len(bandsAn) - 1):
			for orient in range(Nor - 1):
				img11 = self.abs(bandsAn[scale][orient])
				img21 = self.abs(bandsBn[scale][orient])

				for orient2 in range(orient + 1, Nor):
					img13 = self.abs(bandsAn[scale][orient2])
					img23 = self.abs(bandsBn[scale][orient2])
					stsimg2.append(self.compute_cross_term(img11, img13, img21, img23).mean(dim=[1,2,3]))

		return torch.mean(torch.stack(stsimg2), dim=0)

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
		return torch.stack(f)

	def pooling(self, img1, img2):
		tmp = self.compute_L_term(img1, img2) * self.compute_C_term(img1, img2) * self.compute_C01_term(img1, img2) * self.compute_C10_term(img1, img2)
		return torch.mean(tmp**0.25, dim = [1,2,3])

	def compute_L_term(self, img1, img2):
		# expectation over a small window
		mu1 = F.conv2d(img1, self.k)
		mu2 = F.conv2d(img2, self.k)

		Lmap = (2 * mu1 * mu2 + self.C)/( mu1 * mu1 + mu2 * mu2 + self.C)
		return Lmap

	def compute_C_term(self, img1, img2):
		mu1 = F.conv2d(img1, self.k)
		mu2 = F.conv2d(img2, self.k)

		sigma1_sq = F.conv2d(img1**2, self.k) - mu1 * mu1
		sigma1 = torch.sqrt(sigma1_sq)
		sigma2_sq = F.conv2d(img2**2, self.k) - mu2 * mu2
		sigma2 = torch.sqrt(sigma2_sq)

		Cmap = (2*sigma1*sigma2 + self.C)/(sigma1_sq + sigma2_sq + self.C)
		return Cmap

	def compute_C01_term(self, img1, img2):
		win = self.win
		window2 = 1/(win*(win-1)) * np.ones((win,win-1))
		window2 = torch.from_numpy(window2).to(self.device).float().unsqueeze(0).unsqueeze(0)

		img11 = img1[..., :-1]
		img12 = img1[..., 1:]
		img21 = img2[..., :-1]
		img22 = img2[..., 1:]

		mu11 = F.conv2d(img11, window2)
		mu12 = F.conv2d(img12, window2)
		mu21 = F.conv2d(img21, window2)
		mu22 = F.conv2d(img22, window2)

		sigma11_sq = F.conv2d(img11**2, window2) - mu11**2
		sigma12_sq = F.conv2d(img12**2, window2) - mu12**2
		sigma21_sq = F.conv2d(img21**2, window2) - mu21**2
		sigma22_sq = F.conv2d(img22**2, window2) - mu22**2

		sigma1_cross = F.conv2d(img11*img12, window2) - mu11*mu12
		sigma2_cross = F.conv2d(img21*img22, window2) - mu21*mu22


		rho1 = (sigma1_cross + self.C) / (torch.sqrt(sigma11_sq * sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C) / (torch.sqrt(sigma21_sq * sigma22_sq) + self.C)

		C01map = 1 - 0.5*torch.abs(rho1 - rho2)

		return C01map

	def compute_C10_term(self, img1, img2):
		win = self.win
		window2 = 1/(win*(win-1)) * np.ones((win-1,win))
		window2 = torch.from_numpy(window2).to(self.device).float().unsqueeze(0).unsqueeze(0)

		img11 = img1[:,:, :-1, :]
		img12 = img1[:,:, 1: , :]
		img21 = img2[:,:, :-1, :]
		img22 = img2[:,:, 1: , :]

		mu11 = F.conv2d(img11, window2)
		mu12 = F.conv2d(img12, window2)
		mu21 = F.conv2d(img21, window2)
		mu22 = F.conv2d(img22, window2)

		sigma11_sq = F.conv2d(img11**2, window2) - mu11**2
		sigma12_sq = F.conv2d(img12**2, window2) - mu12**2
		sigma21_sq = F.conv2d(img21**2, window2) - mu21**2
		sigma22_sq = F.conv2d(img22**2, window2) - mu22**2

		sigma1_cross = F.conv2d(img11*img12, window2) - mu11*mu12
		sigma2_cross = F.conv2d(img21*img22, window2) - mu21*mu22

		rho1 = (sigma1_cross + self.C)/(torch.sqrt(sigma11_sq)*torch.sqrt(sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C)/(torch.sqrt(sigma21_sq)*torch.sqrt(sigma22_sq) + self.C)

		C10map = 1 - 0.5*torch.abs(rho1 - rho2)

		return C10map

	def compute_cross_term(self, img11, img12, img21, img22):
		# inputs are positive real number
		window2 = 1/(self.win**2)*np.ones((self.win, self.win))
		window2 = torch.from_numpy(window2).to(self.device).float().unsqueeze(0).unsqueeze(0)

		mu11 = F.conv2d(img11, window2)
		mu12 = F.conv2d(img12, window2)
		mu21 = F.conv2d(img21, window2)
		mu22 = F.conv2d(img22, window2)

		sigma11_sq = F.conv2d(img11**2, window2) - mu11**2
		sigma12_sq = F.conv2d(img12**2, window2) - mu12**2
		sigma21_sq = F.conv2d(img21**2, window2) - mu21**2
		sigma22_sq = F.conv2d(img22**2, window2) - mu22**2
		sigma1_cross = F.conv2d(img11*img12, window2) - mu11*mu12
		sigma2_cross = F.conv2d(img21*img22, window2) - mu21*mu22

		rho1 = (sigma1_cross + self.C)/(torch.sqrt(sigma11_sq*sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C)/(torch.sqrt(sigma21_sq*sigma22_sq) + self.C)

		Crossmap = 1 - 0.5*torch.abs(rho1 - rho2)
		return Crossmap
