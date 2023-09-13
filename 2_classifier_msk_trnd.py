import gc
import json
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
import psutil
import sys
import time

import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

from data_loader import load_acc, load_config, load_exp, load_prtn, load_seqs, load_base_model
from data_creator import create_dir, create_exp_labels, create_features_labels, create_model_params, create_pwm_filters, create_pwm_init
from models import model_classifier, load_trained_model
from trainer import trainer_classifier_msk
from utils import create_enhs_configs, create_masks, accuracy_msk
from output import write_msks
from predictor import predictor_msk

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
if 'Genes' in df_exp.columns:
	dict_data['seqs'] = load_seqs(dir_seqs, df_exp['Genes'], n_seq, l_seq, n_cpu)
else:
	raise ValueError('Genes column not present in the expression data')
dict_data['dict_prtn'] = load_prtn(dir_prtn) 
if dict_config['acc'] == 'true':
	dict_data['acc'] = True
	dict_data['data_acc'] = load_acc(dir_acc, df_exp['Genes'], n_seq, n_cells, n_cpu)
arr_genes = np.concatenate([dict_data['dict_prtn']['train'], dict_data['dict_prtn']['valid'], dict_data['dict_prtn']['test']])

dict_config['n_seqs'] = int(dict_config['n_seqs']/dict_config['n_cells'])
dict_features_labels = create_features_labels(dict_data)
if dict_config['conv_init'] != 'true':
	dict_config['n_pwms'], dict_config['l_pwms'], list_tfs, pwm_filters = create_pwm_filters(file_pbms, dir_pbms, pbm_extn)
	list_params = create_model_params(dict_config, pwm_filters, list_tfs)
else:
	list_params = create_model_params(dict_config)

dict_prtn = dict_data['dict_prtn']

del dict_data
gc.collect()
print('Deleted dict_data. Memory Usage: ' + str(psutil.cpu_percent()))

print(json.dumps(dict_config, indent=4))

### Create Log Directories
create_dir(dir_logs + 'msk')
create_dir(dir_logs + 'msk_pred')


### Create Enhancer Masks
file_msks = dir_logs + 'enhancers_masks.txt'
list_msk_configs = create_enhs_configs(n_seq, n_cells)
list_msks = create_masks(list_msk_configs, n_cells)
write_msks(list_msks, file_msks)

### Load trained model
x_param = 0
dict_experiment = list_params[x_param]
dict_experiment['file_trained_model'] = dir_logs + 'trained_model.hdf'
dict_experiment['train_base_model'] = False
dict_experiment['train_pwm_layer'] = False
dict_experiment['train_tf_layer'] = False
trained_model = load_trained_model(dict_experiment)

### Enhancers Iteration Performance
acc_0 = accuracy_msk(trained_model, list_msks[0], dict_features_labels) 
create_dir(dir_logs + 'msk_pred/' + str(0))
dict_experiment['dir_output'] = dir_logs + 'msk_pred/' + str(0)
predictor_msk(dict_experiment, dict_features_labels, dict_prtn, list_msks[0], df_exp, trained_model)
idx = acc_0 < n_cells-1

for i in range(int(len(list_msks)/16) + 1):
# for i in range(1):
	file_enhancer_perf = dir_logs + 'msk/enhancer_performance_' + str(i) + '.txt'
	df_enhancer_perf = pd.DataFrame()
	df_enhancer_perf['Enhancers'] = arr_genes[idx].tolist()
	for j in range(i*16, min((i+1)*16, len(list_msks))):
		msk = list_msks[j]
		df_enhancer_perf['msk_' + str(j)] = accuracy_msk(trained_model, msk, dict_features_labels)[idx].tolist()
		create_dir(dir_logs + 'msk_pred/' + str(j))
		dict_experiment['dir_output'] = dir_logs + 'msk_pred/' + str(j)
		predictor_msk(dict_experiment, dict_features_labels, dict_prtn, msk, df_exp, trained_model)
	df_enhancer_perf.to_csv(file_enhancer_perf, sep='\t', index=False, header=True)
	del df_enhancer_perf
	gc.collect()
	print('Deleted df_enhancer_perf_' + str(i) + '. Memory Usage: ' + str(psutil.cpu_percent()))

print(trained_model.summary())