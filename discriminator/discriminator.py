import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from dataset import *
from ctextgen.model import RNN_VAE

import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)
parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')
parser.add_argument('--mbsize', type=int, default=1024,
                    help='batch size')
parser.add_argument('--h_dim', type=int, default=64,
                    help='model size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9,
                    help='learning rate decay')
parser.add_argument('--n_iter', type=int, default=200000,
                    help='data_size * epoch_size / mbsize')
parser.add_argument('--n_iter_dev', type=int, default=10,
                    help='data_size * epoch_size / mbsize')
parser.add_argument('--log_interval', type=int, default=1000,
                    help='number of epoches to report')
parser.add_argument('--save_interval', type=int, default=5000,
                    help='number of epoches to save a new model')
parser.add_argument('--c_dim', type=int, default=2,
                    help='dim of c')
parser.add_argument('--model_path', type=str, default='./models',
                    help='where to save the model')
parser.add_argument('--checkpoint', type=str, default='./models/disc.bin',
                    help='model checkpoint to use')
parser.add_argument('--test', default=False, action='store_true',
                    help='where to save the model')
args = parser.parse_args()


#------------------------
# Load the dataset
#------------------------
print ("Building dataset...")
dataset = CLS_Dataset(mbsize=args.mbsize)
print ("Dataset built")

#------------------------
# Initiate model
#------------------------
print ("Initiating model...")
model = RNN_VAE(dataset.n_vocab, args.h_dim, args.c_dim, p_word_dropout=0.3, gpu=args.gpu)
print ("Model initiated")

def test(model):
    # Forward proprgation
    num_dev = 0
    acc_dev = 0.0
    for i in range(0, args.n_iter_dev):
        inputs, labels = dataset.next_batch(args.gpu, tag = "dev")
        y_disc_real = model.forward_discriminator(inputs.transpose(0, 1))
	_, est_labels = torch.topk(y_disc_real, 1, dim=1)
	nonzero = torch.nonzero(torch.abs(est_labels.view(-1) - labels))
	wrong = 0 if len(nonzero.size()) == 0 else nonzero.size(0)
	acc_dev += labels.size()[0] - wrong
	num_dev += labels.size()[0]
    num = 0
    acc = 0.0
    for i in range(0, args.n_iter_dev):
        inputs, labels = dataset.next_batch(args.gpu)
        y_disc_real = model.forward_discriminator(inputs.transpose(0, 1))
	_, est_labels = torch.topk(y_disc_real, 1, dim=1)
	nonzero = torch.nonzero(torch.abs(est_labels.view(-1) - labels))
	wrong = 0 if len(nonzero.size()) == 0 else nonzero.size(0)
	acc += labels.size()[0] - wrong
	num += labels.size()[0]
    return acc_dev/num_dev, acc/num

def main():
    trainer_D = optim.Adam(model.discriminator_params, lr=args.lr)

    same_lr_iter = 0
    last_highest_loss = 100.0
    tqdm.monitor_interval = 0

    for it in tqdm(range(args.n_iter)):
        inputs, labels = dataset.next_batch(args.gpu)

	# Forward proprgation
        y_disc_real = model.forward_discriminator(inputs.transpose(0, 1))

	# Loss calculation
        loss_D = F.cross_entropy(y_disc_real, labels)

	# Backward proprgation
        loss_D.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
        trainer_D.step()
        trainer_D.zero_grad()

	# Log
        if it != 0 and it % args.log_interval == 0:
	    acc_dev, acc_train = test(model)
            print('Iter-{}; loss_D: {:.4f}; lr: {:.6f}; acc_dev: {:.4f}; acc_train: {:.4f}'.format(it, float(loss_D), args.lr, acc_dev, acc_train))
	    if float(loss_D) >= last_highest_loss:
	        same_lr_iter += 1
	    else:
		same_lr_iter = 0
		last_highest_loss = float(loss_D)
	    if same_lr_iter > 3:
		args.lr = args.lr * args.lr_decay
		same_lr_iter = 0

	# Model saving
	if it != 0 and it % args.save_interval == 0 and save_model:
	    save_model("_".join(["disc", "loss", str(float(loss_D)), "iter", str(it), "lr", str(args.lr)]) + ".bin")
	    
    print('Final; loss_D: {:.4f}'.format(float(loss_D)))

def save_model(model_name):
    if not os.path.exists(args.model_path):
        os.makedirs()
    torch.save(model, os.path.join(args.model_path, model_name))

if __name__ == '__main__':
    if args.test:
        with open(args.checkpoint, 'rb') as f:
            test_model = torch.load(f)
	    acc_dev, acc_train = test(test_model)    
	    print "acc_dev: ", acc_dev
	    print "acc_train: ", acc_train
    else:
        main()
