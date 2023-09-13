from functools import partial
from itertools import compress
import numpy as np
import os
import pandas as pd
import random
import scipy.stats as ss
import shutil
import sys
import time

from data_loader import load_pbms
from utils import pbm2pwm


def create_exp_labels(df_exp):
	n_genes = len(df_exp)
	n_cols = len(df_exp.columns)
	arr_exp_labels = np.zeros((n_genes, n_cols-1), dtype=float)
	for i_gene in range(n_genes):
		for i_col in range(1, n_cols):
			arr_exp_labels[i_gene, i_col-1] = int(df_exp.iloc[i_gene, i_col])
	print('Labels created. ', arr_exp_labels.shape)
	return arr_exp_labels

def create_features_labels(dict_data):

	seqs = dict_data['seqs']
	exp_labels = dict_data['exp_labels']
	dict_prtn = dict_data['dict_prtn']

	rows_train = dict_prtn['train']
	rows_valid = dict_prtn['valid']
	rows_test = dict_prtn['test']
	n_cells = exp_labels.shape[1]
	arr_layer = (n_cells+1)*np.identity(n_cells+1)

	features_train = seqs[rows_train, :, :, :]
	features_valid = seqs[rows_valid, :, :, :]
	features_test = seqs[rows_test, :, :, :]

	if dict_data['acc']:
		data_acc = dict_data['data_acc']
		arr_layer_train = data_acc[rows_train, :, :]
		arr_layer_valid = data_acc[rows_valid, :, :]
		arr_layer_test = data_acc[rows_test, :, :]
	else:
		arr_layer_train = np.tile(arr_layer, (rows_train.shape[0], 1, 1))
		arr_layer_valid = np.tile(arr_layer, (rows_valid.shape[0], 1, 1))
		arr_layer_test = np.tile(arr_layer, (rows_test.shape[0], 1, 1))

	labels_train = exp_labels[rows_train, :]
	labels_valid = exp_labels[rows_valid, :]	
	labels_test = exp_labels[rows_test, :]

	dict_features_labels={}
	dict_features_labels['features_train'] = features_train
	dict_features_labels['arr_layer_train'] = arr_layer_train
	dict_features_labels['labels_train'] = labels_train
	dict_features_labels['features_valid'] = features_valid
	dict_features_labels['arr_layer_valid'] = arr_layer_valid
	dict_features_labels['labels_valid'] = labels_valid
	dict_features_labels['features_test'] = features_test
	dict_features_labels['arr_layer_test'] = arr_layer_test
	dict_features_labels['labels_test'] = labels_test
	
	print('Partitioned Features and Labels created. ', features_train.shape, arr_layer_train.shape, labels_train.shape, features_valid.shape, arr_layer_valid.shape, labels_valid.shape, features_test.shape, arr_layer_test.shape, labels_test.shape)
	return dict_features_labels

def create_features_labels_flipped(dict_data):

	seqs = dict_data['seqs']
	exp_labels = dict_data['exp_labels']
	dict_prtn = dict_data['dict_prtn_flipped']

	rows_train = dict_prtn['train']
	rows_valid = dict_prtn['valid']
	rows_test = dict_prtn['test']
	n_cells = exp_labels.shape[1]
	arr_layer = (n_cells+1)*np.identity(n_cells+1)

	features_train = seqs[rows_train, :, :, :]
	features_valid = seqs[rows_valid, :, :, :]
	features_test = seqs[rows_test, :, :, :]

	if dict_data['acc']:
		data_acc = dict_data['data_acc']
		arr_layer_train = data_acc[rows_train, :, :]
		arr_layer_valid = data_acc[rows_valid, :, :]
		arr_layer_test = data_acc[rows_test, :, :]
	else:
		arr_layer_train = np.tile(arr_layer, (rows_train.shape[0], 1, 1))
		arr_layer_valid = np.tile(arr_layer, (rows_valid.shape[0], 1, 1))
		arr_layer_test = np.tile(arr_layer, (rows_test.shape[0], 1, 1))

	labels_train = exp_labels[rows_train, :]
	labels_valid = exp_labels[rows_valid, :]	
	labels_test = exp_labels[rows_test, :]

	dict_features_labels={}
	dict_features_labels['features_train'] = features_train
	dict_features_labels['arr_layer_train'] = arr_layer_train
	dict_features_labels['labels_train'] = labels_train
	dict_features_labels['features_valid'] = features_valid
	dict_features_labels['arr_layer_valid'] = arr_layer_valid
	dict_features_labels['labels_valid'] = labels_valid
	dict_features_labels['features_test'] = features_test
	dict_features_labels['arr_layer_test'] = arr_layer_test
	dict_features_labels['labels_test'] = labels_test
	
	print('Partitioned Features and Labels created. ', features_train.shape, arr_layer_train.shape, labels_train.shape, features_valid.shape, arr_layer_valid.shape, labels_valid.shape, features_test.shape, arr_layer_test.shape, labels_test.shape)
	return dict_features_labels

def create_pwm_filters(file_pbms, dir_pbms, pbm_extn):
	dict_pbms = {}
	list_len_pbms = []
	list_tfs = []
	for line in open(file_pbms):
		pbm = line.strip()
		list_tfs.append(pbm)
		dict_pbms[pbm], len_pbm = load_pbms(dir_pbms, pbm, pbm_extn)
		list_len_pbms.append(len_pbm)

	dict_pwms = {}
	max_len = max(list_len_pbms)
	for pbm in dict_pbms.keys():
		dict_pwms[pbm] = pbm2pwm(dict_pbms[pbm], max_len)
	n_pwms = len(dict_pwms)

	pwm_filters = np.zeros([1, 4, max_len, 1, n_pwms])
	i = 0
	for pwm in dict_pwms.keys():
		pwm_filters[:, :, :, 0, i] = dict_pwms[pwm]
		i += 1

	print('PWM Filters created. ', n_pwms, max_len, pwm_filters.shape)
	return n_pwms, max_len, list_tfs, pwm_filters

class create_pwm_init(object):
	def __init__(self, pwm_filters, dict_pwm_prtns):
		self.pwm_filters = pwm_filters
		self.dict_pwm_prtns = dict_pwm_prtns

	def pwms_filter_prtns(self, pwm_filters, dict_pwm_prtns):
		if 'train' in dict_pwm_prtns.keys():
			pwm_filter_train = pwm_filters[:, :, :, :, dict_pwm_prtns['train']]
			if 'fix' in dict_pwm_prtns.keys():
				pwm_filter_fix = pwm_filters[:, :, :, :, dict_pwm_prtns['fix']]
				return pwm_filter_train, pwm_filter_fix
			else:
				return pwm_filter_train
		elif 'fix' in dict_pwm_prtns.keys():
			pwm_filter_fix = pwm_filters[:, :, :, :, dict_pwm_prtns['fix']]
			return pwm_filter_fix
		else:
			return pwm_filters

	def pwms_order(self, dict_pwm_prtns):
		if 'fix' in dict_pwm_prtns.keys() and 'train' in dict_pwm_prtns.keys():
			pwms_order_curr = np.append(dict_pwm_prtns['fix'], dict_pwm_prtns['train'])
			pwms_order_new = []
			for i in range(len(pwms_order_curr)):
				pwms_order_new.append(int(np.where(pwms_order_curr == i)[0][0]))
			return pwms_order_new

def create_model_params(dict_config, pwm_filters=None, list_tfs=None):
	dict_params = []
	if dict_config['conv_init'] == 'true':
		dict_params_i = {}
		dict_params_i['conv_init'] = 'true'
		dict_params_i['n_pwms'] = dict_config['n_pwms']
		dict_params_i['l_pwms'] = dict_config['l_pwms']
		dict_params_i['n_seqs'] = dict_config['n_seqs']
		dict_params_i['l_seqs'] = dict_config['l_seqs']
		dict_params_i['n_cells'] = dict_config['n_cells']
		dict_params_i['cells'] = dict_config['cells']
		dict_params_i['wt_l1'] = dict_config['wt_l1']
		dict_params_i['wt_l2'] = dict_config['wt_l2']
		dict_params_i['wt_kl'] = dict_config['wt_kl']
		dict_params_i['lrng_rt'] = dict_config['lrng_rt']
		dict_params_i['loss_func'] = dict_config['loss_func']
		dict_params_i['dir_logs'] = dict_config['dir_logs']
		dict_params_i['shape_lyrs'] = dict_config['shape_lyrs']
		dict_params_i['act_funcs'] = dict_config['act_funcs']
		dict_params_i['init_seed_value'] = dict_config['init_seed_value']
		dict_params_i['x_train_pwm'] = 'Random_Init'
		dict_params_i['train_pwm'] = 'All'
		dict_params_i['quant_monitor'] = dict_config['quant_monitor']
		dict_params_i['quant_model'] = dict_config['quant_model']
		dict_params_i['n_epochs'] = dict_config['n_epochs']
		dict_params.append(dict_params_i)
	elif dict_config['conv_init'] == 'all':
		dict_params_i = {}
		dict_params_i['conv_init'] = "all"
		dict_params_i['n_pwms'] = dict_config['n_pwms']
		dict_params_i['l_pwms'] = dict_config['l_pwms']
		dict_params_i['n_seqs'] = dict_config['n_seqs']
		dict_params_i['l_seqs'] = dict_config['l_seqs']
		dict_params_i['n_cells'] = dict_config['n_cells']
		dict_params_i['cells'] = dict_config['cells']
		dict_params_i['wt_l1'] = dict_config['wt_l1']
		dict_params_i['wt_l2'] = dict_config['wt_l2']
		dict_params_i['wt_kl'] = dict_config['wt_kl']
		dict_params_i['lrng_rt'] = dict_config['lrng_rt']
		dict_params_i['loss_func'] = dict_config['loss_func']
		dict_params_i['dir_logs'] = dict_config['dir_logs']
		dict_params_i['shape_lyrs'] = dict_config['shape_lyrs']
		dict_params_i['act_funcs'] = dict_config['act_funcs']
		dict_params_i['init_seed_value'] = dict_config['init_seed_value']
		dict_params_i['x_train_pwm'] = 'All_Init'
		dict_params_i['train_pwm'] = 'all'
		dict_params_i['quant_monitor'] = dict_config['quant_monitor']
		dict_params_i['quant_model'] = dict_config['quant_model']
		dict_params_i['n_epochs'] = dict_config['n_epochs']
		id_train_pwms = np.asarray([1]*len(list_tfs))
		pwms_train = np.where(id_train_pwms == 1)[0]
		dict_pwm_prtns={'train':pwms_train}
		pwm_init = create_pwm_init(pwm_filters=pwm_filters, dict_pwm_prtns=dict_pwm_prtns)
		dict_params_i['pwm_filters_train'] = pwm_init.pwms_filter_prtns(pwm_filters, dict_pwm_prtns)
		dict_params.append(dict_params_i)
	elif dict_config['conv_init'] == 'all_true':
		dict_params_i = {}
		dict_params_i['conv_init'] = "all_true"
		dict_params_i['n_pwms'] = dict_config['n_pwms']
		dict_params_i['l_pwms'] = dict_config['l_pwms']
		dict_params_i['n_seqs'] = dict_config['n_seqs']
		dict_params_i['l_seqs'] = dict_config['l_seqs']
		dict_params_i['n_cells'] = dict_config['n_cells']
		dict_params_i['cells'] = dict_config['cells']
		dict_params_i['wt_l1'] = dict_config['wt_l1']
		dict_params_i['wt_l2'] = dict_config['wt_l2']
		dict_params_i['wt_kl'] = dict_config['wt_kl']
		dict_params_i['lrng_rt'] = dict_config['lrng_rt']
		dict_params_i['loss_func'] = dict_config['loss_func']
		dict_params_i['dir_logs'] = dict_config['dir_logs']
		dict_params_i['shape_lyrs'] = dict_config['shape_lyrs']
		dict_params_i['act_funcs'] = dict_config['act_funcs']
		dict_params_i['init_seed_value'] = dict_config['init_seed_value']
		dict_params_i['x_train_pwm'] = 'All_Init'
		dict_params_i['train_pwm'] = 'all'
		dict_params_i['quant_monitor'] = dict_config['quant_monitor']
		dict_params_i['quant_model'] = dict_config['quant_model']
		dict_params_i['n_epochs'] = dict_config['n_epochs']
		id_train_pwms = np.asarray([1]*len(list_tfs))
		pwms_train = np.where(id_train_pwms == 1)[0]
		dict_pwm_prtns={'train':pwms_train}
		pwm_init = create_pwm_init(pwm_filters=pwm_filters, dict_pwm_prtns=dict_pwm_prtns)
		dict_params_i['pwm_filters_train'] = pwm_init.pwms_filter_prtns(pwm_filters, dict_pwm_prtns)
		dict_params.append(dict_params_i)
	elif dict_config['conv_init'] == 'none':
		dict_params_i = {}
		dict_params_i['conv_init'] = "none"
		dict_params_i['n_pwms'] = dict_config['n_pwms']
		dict_params_i['l_pwms'] = dict_config['l_pwms']
		dict_params_i['n_seqs'] = dict_config['n_seqs']
		dict_params_i['l_seqs'] = dict_config['l_seqs']
		dict_params_i['n_cells'] = dict_config['n_cells']
		dict_params_i['cells'] = dict_config['cells']
		dict_params_i['wt_l1'] = dict_config['wt_l1']
		dict_params_i['wt_l2'] = dict_config['wt_l2']
		dict_params_i['wt_kl'] = dict_config['wt_kl']
		dict_params_i['lrng_rt'] = dict_config['lrng_rt']
		dict_params_i['loss_func'] = dict_config['loss_func']
		dict_params_i['dir_logs'] = dict_config['dir_logs']
		dict_params_i['shape_lyrs'] = dict_config['shape_lyrs']
		dict_params_i['act_funcs'] = dict_config['act_funcs']
		dict_params_i['init_seed_value'] = dict_config['init_seed_value']
		dict_params_i['x_train_pwm'] = 'All_Init'
		dict_params_i['train_pwm'] = 'none'
		dict_params_i['quant_monitor'] = dict_config['quant_monitor']
		dict_params_i['quant_model'] = dict_config['quant_model']
		dict_params_i['n_epochs'] = dict_config['n_epochs']
		id_train_pwms = np.asarray([0]*len(list_tfs))
		pwms_fix = np.where(id_train_pwms == 0)[0]
		dict_pwm_prtns={'fix':pwms_fix}
		pwm_init = create_pwm_init(pwm_filters=pwm_filters, dict_pwm_prtns=dict_pwm_prtns)
		dict_params_i['pwm_filters_fix'] = pwm_init.pwms_filter_prtns(pwm_filters, dict_pwm_prtns)
		dict_params.append(dict_params_i)
	else:
		n_tfs = len(list_tfs)
		for i in range(n_tfs + 1):
			dict_params_i = {}
			dict_params_i['conv_init'] = False
			dict_params_i['n_pwms'] = dict_config['n_pwms']
			dict_params_i['l_pwms'] = dict_config['l_pwms']
			dict_params_i['n_seqs'] = dict_config['n_seqs']
			dict_params_i['l_seqs'] = dict_config['l_seqs']
			dict_params_i['n_cells'] = dict_config['n_cells']
			dict_params_i['cells'] = dict_config['cells']
			dict_params_i['wt_l1'] = dict_config['wt_l1']
			dict_params_i['wt_l2'] = dict_config['wt_l2']
			dict_params_i['wt_kl'] = dict_config['wt_kl']
			dict_params_i['lrng_rt'] = dict_config['lrng_rt']
			dict_params_i['loss_func'] = dict_config['loss_func']
			dict_params_i['dir_logs'] = dict_config['dir_logs']
			dict_params_i['shape_lyrs'] = dict_config['shape_lyrs']
			dict_params_i['act_funcs'] = dict_config['act_funcs']
			dict_params_i['init_seed_value'] = dict_config['init_seed_value']
			dict_params_i['x_train_pwm'] = i
			dict_params_i['quant_monitor'] = dict_config['quant_monitor']
			dict_params_i['n_epochs'] = dict_config['n_epochs']
			if i < n_tfs:
				id_train_pwms = np.asarray([0]*i + [1] + [0]*(n_tfs-1-i))
				dict_params_i['train_pwm'] = list_tfs[i]
			else:
				id_train_pwms = np.asarray([0]*i)
				dict_params_i['train_pwm'] = 'None'
			pwms_train = np.where(id_train_pwms == 1)[0]
			pwms_fix = np.where(id_train_pwms == 0)[0]
			dict_pwm_prtns={'train':pwms_train, 'fix':pwms_fix}
			pwm_init = create_pwm_init(pwm_filters=pwm_filters, dict_pwm_prtns=dict_pwm_prtns)
			dict_params_i['pwm_filters_train'], dict_params_i['pwm_filters_fix'] = pwm_init.pwms_filter_prtns(pwm_filters, dict_pwm_prtns)
			dict_params_i['pwms_order'] = pwm_init.pwms_order(dict_pwm_prtns)
			dict_params.append(dict_params_i)
	print('dict_params created.')
	return dict_params

def create_dir(dir_logs):
	if os.path.exists(dir_logs):
		shutil.rmtree(dir_logs)
	os.mkdir(dir_logs)
	print(dir_logs  + ' created.')


def create_pwms_select(file_pwms_select):
	list_pwms_select = []
	for line in open(file_pwms_select):
		trans = line.strip()
		list_pwms_select.append(int(trans[5:]))
	return list_pwms_select