#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class LossFunction(nn.Module):
	def __init__(self, gpu, **kwargs):
		super(LossFunction, self).__init__()
		self.tau = nn.Parameter(data=torch.tensor(0.07), requires_grad=True) if kwargs['tau_equi'] == 0 else kwargs['tau_equi']
		self.labels = None
		self.masks = None
		self.last_local_batch_size = None

		if gpu == 0:
			print('Initialised SimCLR Loss')

	def cross_entropy_loss(self, x, y):
		sam_num = x.shape[0]

		delta = 1e-7
		x = x + delta
		log_x = torch.log(x)[:,:sam_num]
		
		return -torch.sum(torch.diagonal(log_x))

		# return -torch.sum(torch.matmul(y, torch.log(x)))

	def softmax(self, a):
		eps = 1e-7
		exp_a = torch.exp(a)
		sum_exp_a = torch.sum(exp_a,dim=1) + eps
		y = exp_a / (sum_exp_a.unsqueeze(1))

		return y

	def forward(self, z_equi_t, z_aug):
		q_a = z_equi_t
		q_b = z_aug

		q_a = F.normalize(q_a, dim=-1, p=2)
		q_b = F.normalize(q_b, dim=-1, p=2)

		local_batch_size = q_a.size(0)

		k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

		if local_batch_size != self.last_local_batch_size:
			self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
			total_batch_size = local_batch_size * utils.get_world_size()
			self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
			self.last_local_batch_size = local_batch_size

		logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
		logits_aa = logits_aa - self.masks
		logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
		logits_bb = logits_bb - self.masks
		logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
		logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

		# soft_a = self.softmax(torch.cat([logits_ab, logits_aa], dim=1))
		# soft_b = self.softmax(torch.cat([logits_ba, logits_bb], dim=1))

		# loss_a = self.cross_entropy_loss(soft_a, self.labels)
		# loss_b = self.cross_entropy_loss(soft_b, self.labels)

		loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
		loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)

		loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
		with torch.no_grad():
			pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
			correct = pred.eq(self.labels).sum()
			acc = 100 * correct / local_batch_size
		
		return {'loss': loss, 'equi_loss': loss, 'equi_acc': acc, 'acc': acc}