from functools import partial
from itertools import compress
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import sharedctypes
import numpy as np
import os
import pandas as pd
import random
import scipy.stats as ss
import shutil
import sys
import time

import tensorflow as tf
from tensorflow import keras

from utils import nucleotide2ind

config_int = ['n_seqs', 'l_seqs', 'n_cells', 'n_pwms', 'l_pwms', 'n_cpu', 'init_seed_value', 'n_epochs', 'n_grid_start', 'n_grid_end']
config_flt = ['wt_l1', 'wt_l2', 'wt_kl', 'lrng_rt']
config_int_list = ['shape_lyrs', 'trials']
config_flt_list = []
config_dir = ['dir_data', 'dir_prtn', 'dir_seqs', 'dir_acc', 'dir_msks', 'dir_pbms', 'dir_logs', 'dir_trials']
config_str = ['file_gene_exp', 'file_pbms','loss_func', 'acc', 'pbm_extn', 'pbm_train', 'conv_init', 'quant_monitor', 'quant_model']

def load_config(file_config):
	dict_config = {}
	for line in open(file_config):
		trans = line.strip().split()
		if trans[0] in config_int:
			dict_config[trans[0]] = int(trans[1])
		elif trans[0] in config_flt:
			dict_config[trans[0]] = float(trans[1])
		elif trans[0] in config_int_list:
			if len(trans) > 1:
				dict_config[trans[0]] = [int(x) for x in trans[1:]]
			else:
				dict_config[trans[0]] = []
		elif trans[0] in config_flt_list:
			dict_config[trans[0]] = [flt(x) for x in trans[1:]]
		elif trans[0] in config_dir:
			dict_config[trans[0]] = trans[1] + '/'
		elif trans[0] in config_str:
			dict_config[trans[0]] = str(trans[1])
		else:
			dict_config[trans[0]] = trans[1:]
	return dict_config

def load_exp(file_exp):
	df_exp = pd.read_csv(file_exp, sep='\t')
	print('Expression file loaded. ', df_exp.shape)
	return df_exp

def load_gene_seqs(dir_seqs, id_gene, n_seq, len_seq):
	j_f = 0
	enhancer = True
	gene_seqs = np.zeros((2, 4*n_seq, len_seq))
	for index, line in enumerate(open(dir_seqs + id_gene + '.fa')):
		line = line.strip()
		if line[0] == ">":
			if "Dummy" in line.strip():
				enhancer = False
				j_f += 1
			else:
				enhancer = True
		elif line[0] != ">" and enhancer:
			for k_f in range(len_seq):
				ntnum = nucleotide2ind(line[k_f])
				if ntnum in [0, 1, 2, 3]:
					# gene_seqs[j_f][0][ntnum][k_f] = 1
					# gene_seqs[j_f][1][3 - ntnum][len_seq - 1 - k_f] = 1
					gene_seqs[0][4*j_f+ntnum][k_f] = 1
					gene_seqs[1][4*j_f+3 - ntnum][len_seq - 1 - k_f] = 1
			j_f += 1
	return gene_seqs

def load_gene_seqs_msk(dir_seqs, id_gene, n_seq, len_seq, msk):
	j_f = 0
	msk_seq = np.repeat(msk, 4).tolist()
	enhancer = True
	gene_seqs = np.zeros((2, 4*n_seq, len_seq))
	for index, line in enumerate(open(dir_seqs + id_gene + '.fa')):
		line = line.strip()
		if line[0] == ">":
			if "Dummy" in line.strip():
				enhancer = False
				j_f += 1
			else:
				enhancer = True
		elif line[0] != ">" and enhancer:
			for k_f in range(len_seq):
				ntnum = nucleotide2ind(line[k_f])
				if ntnum in [0, 1, 2, 3]:
					# gene_seqs[j_f][0][ntnum][k_f] = 1
					# gene_seqs[j_f][1][3 - ntnum][len_seq - 1 - k_f] = 1
					gene_seqs[0][4*j_f+ntnum][k_f] = 1
					gene_seqs[1][4*j_f+3 - ntnum][len_seq - 1 - k_f] = 1
			j_f += 1
	return gene_seqs[:, msk_seq, :]


def load_seqs(dir_seqs, list_genes, n_seq, len_seq, n_cpu):
	start = time.time()
	seqs = []
	n_process = int(np.ceil(len(list_genes)/n_cpu))
	for i in range(n_process):
		pool_process = Pool(n_cpu)
		seqs = seqs + pool_process.starmap(load_gene_seqs, [(dir_seqs, list_genes[x], n_seq, len_seq) for x in range(i*n_cpu, min((i+1)*n_cpu, len(list_genes)))])
		pool_process.close()
		pool_process.join()
	seqs = np.asarray(seqs)
	print('Sequences loaded. ', seqs.shape, round((time.time() - start)/60, 3))
	return seqs

def load_seqs_msk(dir_seqs, list_genes, arr_msks, n_seq, len_seq, n_cpu):
	start = time.time()
	seqs = []
	n_process = int(np.ceil(len(list_genes)/n_cpu))
	for i in range(n_process):
		pool_process = Pool(n_cpu)
		seqs = seqs + pool_process.starmap(load_gene_seqs_msk, [(dir_seqs, list_genes[x], n_seq, len_seq, arr_msks[x]) for x in range(i*n_cpu, min((i+1)*n_cpu, len(list_genes)))])
		pool_process.close()
		pool_process.join()
	seqs = np.asarray(seqs)
	print('Sequences loaded. ', seqs.shape, round((time.time() - start)/60, 3))
	return seqs

def load_gene_acc(dir_acc, id_gene, n_seqs, n_cells):
	gene_acc = np.zeros((n_seqs, n_cells))
	i_a = 0
	for line in open(dir_acc + id_gene + '.txt'):
		line = line.strip()
		if line[0] != '>':
			trans = line.split()
			for j_a in range(len(trans)):
				gene_acc[i_a][j_a] = float(trans[j_a])
			i_a += 1
	return gene_acc

def load_gene_acc_msk(dir_acc, id_gene, n_seqs, n_cells, msk):
	gene_acc = np.zeros((n_seqs, n_cells))
	i_a = 0
	for line in open(dir_acc + id_gene + '.txt'):
		line = line.strip()
		if line[0] != '>':
			trans = line.split()
			for j_a in range(len(trans)):
				gene_acc[i_a][j_a] = float(trans[j_a])
			i_a += 1
	return gene_acc[msk, :]

def load_acc(dir_acc, list_genes, n_seqs, n_cells, n_cpu):
	start = time.time()
	acc = []
	n_process = int(np.ceil(len(list_genes)/n_cpu))
	for i in range(n_process):
		pool_process = Pool(n_cpu)
		acc = acc + pool_process.starmap(load_gene_acc, [(dir_acc, list_genes[x], n_seqs, n_cells) for x in range(i*n_cpu, min((i+1)*n_cpu, len(list_genes)))])
		pool_process.close()
		pool_process.join()
	acc = np.asarray(acc)
	print('Accessibility data loaded. ', acc.shape, round((time.time() - start)/60, 3))
	return acc

def load_acc_msk(dir_acc, list_genes, arr_msks, n_seqs, n_cells, n_cpu):
	start = time.time()
	acc = []
	n_process = int(np.ceil(len(list_genes)/n_cpu))
	for i in range(n_process):
		pool_process = Pool(n_cpu)
		acc = acc + pool_process.starmap(load_gene_acc_msk, [(dir_acc, list_genes[x], n_seqs, n_cells, arr_msks[x]) for x in range(i*n_cpu, min((i+1)*n_cpu, len(list_genes)))])
		pool_process.close()
		pool_process.join()
	acc = np.asarray(acc)
	print('Accessibility data loaded. ', acc.shape, round((time.time() - start)/60, 3))
	return acc

def load_gene_msk(dir_msk, id_gene, n_seqs):
	gene_msk = np.zeros((n_seqs), dtype=bool)
	i_a = 0
	for line in open(dir_msk + id_gene + '.txt'):
		line = line.strip()
		if line[0] != '>':
			trans = line.split()
			gene_msk[i_a] = bool(int(trans[0]) == 1)
			i_a += 1
	return gene_msk

def load_msk(dir_msk, list_genes, n_seqs, n_cpu):
	start = time.time()
	msk = []
	n_process = int(np.ceil(len(list_genes)/n_cpu))
	for i in range(n_process):
		pool_process = Pool(n_cpu)
		msk = msk + pool_process.starmap(load_gene_msk, [(dir_msk, list_genes[x], n_seqs) for x in range(i*n_cpu, min((i+1)*n_cpu, len(list_genes)))])
		pool_process.close()
		pool_process.join()
	msk = np.asarray(msk)
	print('Mask data loaded. ', msk.shape, round((time.time() - start)/60, 3))
	return msk.tolist()

def load_prtn(dir_prtn):
	dict_prtn = {}
	for typ_prtn in ['train', 'valid', 'test']:
		dict_prtn[typ_prtn] = np.loadtxt(dir_prtn + typ_prtn + '.txt', dtype=int)
	print('Partitions loaded. ', [dict_prtn[typ_prtn].shape for typ_prtn in ['train', 'valid', 'test']])
	return dict_prtn

def load_prtn_flipped(dir_prtn):
	dict_prtn = {}
	for typ_prtn in ['train_flipped', 'valid_flipped', 'test_flipped']:
		dict_prtn[typ_prtn.split('_')[0]] = np.loadtxt(dir_prtn + typ_prtn + '.txt', dtype=int)
	print('Partitions loaded. ', [dict_prtn[typ_prtn].shape for typ_prtn in ['train', 'valid', 'test']])
	return dict_prtn

def load_pbms(dir_pwms, pbm, pbm_extn=".wtmx"):
	list_pbms = []
	for i, line in enumerate(open(dir_pwms + "/" + pbm + '.' + pbm_extn)):
		trans = line.strip().split()
		if i > 0 and trans[0] != "<":
			list_pbms.append([float(x) for x in trans])
	return list_pbms, len(list_pbms)

def load_pred(dir_pred):
    df_pred = pd.DataFrame()
    for typ in ["train", "valid", "test"]:
        file_pred_typ = dir_pred + "pred_" + typ + ".txt"
        df_pred_typ = pd.read_csv(file_pred_typ, sep="\t")
        df_pred = pd.concat([df_pred, df_pred_typ], axis=0)
        df_pred.index = range(len(df_pred))
    return df_pred



##### Redundant #####
def load_base_model(file_base_model, train_base_model, custom_loss_func=None):
	if custom_loss_func:
		base_model = keras.models.load_model(file_base_model, custom_objects={"loss": custom_loss_func})
	else:
		base_model = keras.models.load_model(file_base_model)
	base_model.trainable = train_base_model
	return base_model
