import json
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
import sys
import time

import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

from data_loader import load_acc, load_config, load_exp, load_prtn, load_seqs, load_base_model
from data_creator import create_dir, create_exp_labels, create_features_labels, create_model_params, create_pwm_filters, create_pwm_init
from models import model_classifier
from trainer import trainer_classifier
from predictor import predictor_raw
from output import write_pwms

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
dict_features_labels = create_features_labels(dict_data)

if dict_config['conv_init'] != 'true':
	dict_config['n_pwms'], dict_config['l_pwms'], list_tfs, pwm_filters = create_pwm_filters(file_pbms, dir_pbms, pbm_extn)
	file_fltrs_init = dir_logs + 'fltrs_init.txt'
	write_pwms(pwm_filters, file_fltrs_init)
	list_params = create_model_params(dict_config, pwm_filters, list_tfs)
	n_pwms = dict_config["n_pwms"]
else:
	list_params = create_model_params(dict_config)

print(json.dumps(dict_config, indent=4))

### Create Log Directories
create_dir(dir_logs + 'training_plots')

### Performance
file_perf = dir_logs + 'performance.txt'
df_perf = pd.DataFrame(columns=['X_Params', 'PWM'] + ['train_mse_' + cell for cell in cells[1:]] + ['train_mse'] + ['valid_mse_' + cell for cell in cells[1:]] + ['valid_mse'] + ['test_mse_' + cell for cell in cells[1:]] + ['test_mse', 'RunTime'])
# n_process = int(np.ceil(len(dict_params)/n_cpu))
# results_pool = []
# for i in range(n_process-2, n_process):
# 	pool_process = Pool(n_cpu)
# 	results_pool = results_pool + pool_process.starmap(trainer_classifier, [(dict_params, dict_features_labels, base_model, x_param) for x_param in range(i*n_cpu, min((i+1)*n_cpu, len(dict_params)))])
# 	pool_process.close()
# 	pool_process.join()

# for res in results_pool:
# 	x_param = res[0]
# 	df_perf.loc[x_param] = res
# df_perf.to_csv(file_perf, sep='\t', index=False, header=True)

x_param = 0
dict_experiment = list_params[x_param]
dict_experiment['dir_output'] = dir_logs
res = trainer_classifier(dict_experiment, dict_features_labels)
df_perf.loc[0] = res
df_perf.to_csv(file_perf, sep='\t', index=False, header=True)
predictor_raw(dict_experiment, dict_features_labels, dict_data['dict_prtn'], df_exp)

model_sum=model_classifier(list_params[x_param])
print(model_sum.summary())
print(res)
