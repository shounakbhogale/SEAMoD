import gc
import json
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import psutil
import shutil
import sys
import time

import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

from data_loader import load_acc_msk, load_config, load_exp, load_prtn, load_seqs_msk, load_base_model, load_msk, load_pred
from data_creator import create_dir, create_exp_labels, create_features_labels, create_model_params, create_pwm_filters, create_pwm_init
from models import model_classifier, load_trained_model
from trainer import pre_trainer_classifier, trainer_classifier
from utils import create_enhs_configs, create_masks, accuracy_msk
from output import write_msks, write_pwms
from predictor import predictor_raw

### Read cmd line inputs
dict_cmnd_input = {}
for i in range(len(sys.argv)):
	if sys.argv[i] == '-config':
			dict_cmnd_input['config'] = sys.argv[i+1]

### Load config file
dict_config = load_config(dict_cmnd_input['config'])
dir_logs = dict_config["dir_logs"]
dir_pbms = dict_config["dir_data"] + dict_config["dir_pbms"]
dir_prtn = dict_config["dir_data"] + dict_config["dir_prtn"]
dir_seqs = dict_config["dir_data"] + dict_config["dir_seqs"]
dir_acc = dict_config["dir_data"] + dict_config["dir_acc"]
dir_msk = dict_config["dir_logs"] + dict_config["dir_msks"]
file_exp = dir_prtn + dict_config["file_gene_exp"]
file_pbms = dict_config["dir_logs"] + dict_config['file_pbms']
l_seq = dict_config['l_seqs']
pbm_extn = dict_config['pbm_extn']
n_cpu = dict_config['n_cpu']
n_cells = dict_config['n_cells']
n_seq = dict_config['n_seqs']
cells = dict_config['cells']

### Load Data and created features and labels 
dict_data = {}
dict_data['acc'] = False
df_exp = load_exp(file_exp)
dict_data['exp_labels'] = create_exp_labels(df_exp)
arr_msks = load_msk(dir_msk, df_exp['Genes'], n_seq, n_cpu)
if 'Genes' in df_exp.columns:
	dict_data['seqs'] = load_seqs_msk(dir_seqs, df_exp['Genes'], arr_msks, n_seq, l_seq, n_cpu)
else:
	raise ValueError('Genes column not present in the expression data')
dict_data['dict_prtn'] = load_prtn(dir_prtn) 
if dict_config['acc'] == 'true':
	dict_data['acc'] = True
	dict_data['data_acc'] = load_acc_msk(dir_acc, df_exp['Genes'], arr_msks, n_seq, n_cells, n_cpu)
arr_genes = np.concatenate([dict_data['dict_prtn']['train'], dict_data['dict_prtn']['valid'], dict_data['dict_prtn']['test']])

dict_config['n_seqs'] = int(dict_config['n_seqs']/dict_config['n_cells'])
dict_features_labels = create_features_labels(dict_data)
if dict_config['conv_init'] != 'true':
	dict_config['n_pwms'], dict_config['l_pwms'], list_tfs, pwm_filters = create_pwm_filters(file_pbms, dir_pbms, pbm_extn)
	list_params = create_model_params(dict_config, pwm_filters, list_tfs)
else:
	list_params = create_model_params(dict_config)

n_pwms = dict_config["n_pwms"]
dict_prtn = dict_data['dict_prtn']

del dict_data
gc.collect()
print('Deleted dict_data. Memory Usage: ' + str(psutil.cpu_percent()))

print(json.dumps(dict_config, indent=4))

### Load pretrained model
x_param = 0
dict_experiment = list_params[x_param]
dict_experiment['file_trained_model'] = dir_logs + 'msk_best/trained_model.hdf'
dict_experiment['train_base_model'] = False
dict_experiment['train_pwm_layer'] = False
dict_experiment['train_tf_layer'] = False

### Performance
file_perf = dir_logs + 'performance_msk_best_train_kdtf.txt'
df_perf = pd.DataFrame(columns=['X_Params', 'PWM'] + ['train_mse_' + cell for cell in cells[1:]] + ['train_mse'] + ['valid_mse_' + cell for cell in cells[1:]] + ['valid_mse'] + ['test_mse_' + cell for cell in cells[1:]] + ['test_mse', 'RunTime', 'kd_TF', 'TFs'])

dir_kdtf = dir_logs + 'msk_best_train_kdtf'
if os.path.exists(dir_kdtf):
	shutil.rmtree(dir_kdtf)
os.mkdir(dir_kdtf)
for i in range(n_pwms):
	dir_output = dir_kdtf + '/pwms_' + str(i)
	file_fltrs = dir_output + '/fltrs_trnd.wtmx'
	os.mkdir(dir_output)
	trained_model = load_trained_model(dict_experiment)
	trained_model_wts = trained_model.get_weights()
	trained_model_wts[1][i] = 0
	trained_model.set_weights(trained_model_wts)
	dict_experiment['trained_model'] = trained_model
	dict_experiment['dir_output'] = dir_output	
	write_pwms(trained_model.get_weights()[0], file_fltrs)
	predictor_raw(dict_experiment, dict_features_labels, dict_prtn, df_exp, trained_model)

	file_fltr_wts = dir_output + '/fltr_wts.txt'
	model_fltr_wts = trained_model.get_weights()[1]
	df_fltr_wts = pd.DataFrame()
	df_fltr_wts['PWMs'] = ['pwms_' + str(x) for x in range(n_pwms)]
	for j in range(len(cells)):
		df_fltr_wts[cells[j]] = model_fltr_wts[:, 0, 0, j]
	df_fltr_wts.to_csv(file_fltr_wts, sep='\t', index=False, header=True)

	res = pre_trainer_classifier(dict_experiment, dict_features_labels)
	if dict_config['conv_init'] != 'true':
		df_perf.loc[i] = res + ['pwms_' + str(i), list_tfs[i]]
	else:
		df_perf.loc[i] = res + ['pwms_' + str(i), 'Random']

dir_output = dir_kdtf + '/WT'
os.mkdir(dir_output)
dict_experiment['dir_output'] = dir_output
trained_model = load_trained_model(dict_experiment)
dict_experiment['trained_model'] = load_trained_model(dict_experiment)

res = pre_trainer_classifier(dict_experiment, dict_features_labels)	
predictor_raw(dict_experiment, dict_features_labels, dict_prtn, df_exp, trained_model)

file_fltr_wts = dir_output + '/fltr_wts.txt'
model_fltr_wts = trained_model.get_weights()[1]
df_fltr_wts = pd.DataFrame()
df_fltr_wts['PWMs'] = ['pwms_' + str(x) for x in range(n_pwms)]
for j in range(len(cells)):
	df_fltr_wts[cells[j]] = model_fltr_wts[:, 0, 0, j]
df_fltr_wts.to_csv(file_fltr_wts, sep='\t', index=False, header=True)
df_perf.loc[n_pwms] = res + ['WT', 'None']

df_perf.to_csv(file_perf, sep='\t', index=False, header=True)


#### Accuracy Analysis
file_perf_kdtf = dir_logs + "msk_best/analysis/kdtf_perf.txt"
df_perf_kdtf = pd.DataFrame(columns=["TF"] + cells[1:] + ["Total"])

file_perf_acc = dir_logs + "msk_best/analysis/accuracy.txt"
df_perf_acc = pd.read_csv(file_perf_acc, sep="\t")

df_pred_wt = load_pred(dir_kdtf + "/WT/")
row_perf_wt = []
for cell in cells[1:]:
    t_up = df_perf_acc.loc[0, cell]
    t_down = df_perf_acc.loc[1, cell]
    n_up_true = sum((df_pred_wt[cell] == 1) & (df_pred_wt["pred_" + cell] > t_up))
    n_nc_true = sum(
        (df_pred_wt[cell] == 0)
        & ((df_pred_wt["pred_" + cell] <= t_up) & (df_pred_wt["pred_" + cell] >= t_down))
    )
    n_down_true = sum((df_pred_wt[cell] == -1) & (df_pred_wt["pred_" + cell] < t_down))
    acc = (n_up_true + n_nc_true + n_down_true) / len(df_pred_wt)
    row_perf_wt.append(acc)
arr_perf_wt = np.array(row_perf_wt + [np.mean(row_perf_wt)])

for i in range(n_pwms):
	dir_kd = dir_kdtf + "/pwms_" + str(i) + "/"
	df_pred = load_pred(dir_kd)
	row_perf_kdtf = []
	for cell in cells[1:]:
		t_up = df_perf_acc.loc[0, cell]
		t_down = df_perf_acc.loc[1, cell]
		n_up_true = sum((df_pred[cell] == 1) & (df_pred["pred_" + cell] > t_up))
		n_nc_true = sum(
		    (df_pred[cell] == 0)
		    & ((df_pred["pred_" + cell] <= t_up) & (df_pred["pred_" + cell] >= t_down))
		)
		n_down_true = sum((df_pred[cell] == -1) & (df_pred["pred_" + cell] < t_down))
		acc = (n_up_true + n_nc_true + n_down_true) / len(df_pred)
		row_perf_kdtf.append(acc)
	df_perf_kdtf.loc[i-1] = ['pwms_' + str(i)] + list(np.array(row_perf_kdtf + [np.mean(row_perf_kdtf)]) - arr_perf_wt)
df_perf_kdtf.to_csv(file_perf_kdtf, sep='\t', index=False, header=True)