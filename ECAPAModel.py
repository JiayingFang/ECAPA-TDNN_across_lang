'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle, glob
from itertools import cycle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN

#######################
## Edited by Jiaying ##
#######################
# Part of this code from Dr. Weiwei Lin
SIGMAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
          1e-1, 1, 5, 10, 15, 20, 25,
          30, 35, 100, 1e3, 1e4, 1e5, 1e6)

def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res
def gaussian_kernel(x, y, sigmas):
    sigmas = torch.tensor(sigmas, device=x.get_device())
    beta = 1. / (2. * sigmas[:, None, None])
    dist = my_cdist(x, y)[:, None, None]
    s = -beta * dist
    return s.exp().mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).to(device)
		#######################
		## Edited by Jiaying ##
		#######################
		# if uses DataParallel, uncomment the line below
		# self.speaker_encoder = nn.DataParallel(ECAPA_TDNN(C = C)).to(device)
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).to(device)

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def mmd(self, x, y, sigmas=SIGMAS):
		cost = gaussian_kernel(x, x, sigmas).to(device) \
			+ gaussian_kernel(y, y, sigmas).to(device) \
			- 2 * gaussian_kernel(x, y, sigmas).to(device)
		return cost
  
	def train_network(self, epoch, source_loader, target_loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		dataloader_iterator = iter(target_loader)

		# Iterate over two data loaders
		#######################
		## Edited by Jiaying ##
		#######################
		for num, data1 in enumerate(source_loader):

			try:
				data2 = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(target_loader)
				data2 = next(dataloader_iterator)
			self.zero_grad()
			data, labels = data1
			tgt = data2
			labels            = torch.LongTensor(labels).to(device)
			speaker_embedding, segment_src = self.speaker_encoder.forward(data.to(device), aug = True)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)
			embed_tgt, segment_tgt = self.speaker_encoder.forward(tgt.to(device), aug = True)
			ndomain_loss = 100*self.mmd(embed_tgt, speaker_embedding).to(device)
			ndomain_loss2 = 100*self.mmd(segment_tgt, segment_src).to(device)
			# print("nloss:", nloss)
			# print("ndomain_loss:", ndomain_loss)
			nloss = nloss + ndomain_loss + ndomain_loss2			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / source_loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)


	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[0])
			files.append(line.split()[1])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to(device)

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).to(device)
			# Speaker embeddings
			with torch.no_grad():
				embedding_1, sg1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2, sg2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[0]]
			embedding_21, embedding_22 = embeddings[line.split()[1]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[2]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		# for name, param in self_state.items():
		# 	print(name)
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				#######################
				## Edited by Jiaying ##
				#######################
				# If uses data parallel, uncomment the line below
				# name = name.replace("speaker_encoder.", "speaker_encoder.module.")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)