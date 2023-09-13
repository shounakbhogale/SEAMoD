import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def nucleotide2ind(char):
	switcher = {
		"A": 0,
		"a": 0,
		"C": 1,
		"c": 1,
		"G": 2,
		"g": 2,
		"T": 3,
		"t": 3,
	}
	return switcher.get(char)

def pbm2pwm(pbm, max_len):
	arr_pwms = []
	for i in range(len(pbm)):
		temp = []
		for j in range(4):
			if pbm[i][j] == 0:
				temp.append(-10)
			else:
				temp.append(np.log2(4 * pbm[i][j] / sum(pbm[i])))
		arr_pwms.append(temp)
	for i in range(len(pbm), max_len):
		arr_pwms.append([0, 0, 0, 0])
	return np.array(arr_pwms).T

### Convolutional Filter Initialization
class kernel_pwm_init(tf.keras.initializers.Initializer):
    def __init__(self, pwms_init):
        self.pwm = pwms_init

    def __call__(self, shape, dtype=None):
        return self.pwm

    def get_config(self): 
        return {"pwms_init": self.pwm}


### Enhancer Iteration Method
def create_enhs_configs(n_seqs, n_cells):
	n_seqs_cell = int(n_seqs/n_cells)
	list_configs = [[True] + [False]*(n_seqs_cell-1)]
	for i in range(1, n_seqs_cell):
		list_configs.append([False]*(i) + [True] + [False]*(n_seqs_cell-i-1))
	return list_configs

def create_masks(list_configs, len_msk):
	list_msks = []
	if len_msk > 1:
		for i in range(len(list_configs)):
			list_msks = list_msks + [list_configs[i] + x for x in create_masks(list_configs, len_msk-1)]
	else:
		list_msks = list_configs
	return list_msks

### Loss Functions

# kl stands for Kullback–Leibler divergence. It should only be used if the convolutional filters should resemble known PWMs. This loss tries to minimize the divergence between the known PMWs and convolutional filters. 

def kl_divergence(x, pwm): # KL Divergence between x (tensor comrpised of convolutional filters) and pwm (tensor comprised of all known pwms). 
    return 0.5*K.sum(K.exp(pwm)*(pwm-x)/(K.sum(K.exp(pwm), axis=2) + 1e-6)) + 0.5*K.sum(K.exp(x)*(x-pwm)/(K.sum(K.exp(x), axis=2) + 1e-6))

def kl_mse(x, pwm, wt_kl): # MSE Loss + KL divergece Loss
    def loss(y_true, y_pred):
        return K.mean(K.square(y_true-y_pred)) + wt_kl*kl_divergence(x, pwm)
    return loss

def degs_mse(): # MSE Loss with DEGs only
	def loss(y_true, y_pred):
		return K.sum(K.square((y_true-y_pred)*y_true))/(1e-6 + K.sum(K.square(y_true)))
	return loss

def custom_loss_func(loss_func, x=None, pwm=None, wt_kl=None):
    if loss_func == "mean_squared_error_kl":
        return kl_mse(x, pwm, wt_kl)
    elif loss_func == 'mean_squared_error_degs':
    	return degs_mse()
    else:
    	return loss_func


### PWM Layer Regularization
def custom_regularizer_kl(pwm, wt_kl=1):
	pwm = tf.cast(pwm, tf.float32)
	def custom_reg(weights):
		weigths = tf.cast(weights, tf.float32)
		w_p = 0.5*K.sum(K.exp(weights)*(weights-pwm)/(K.sum(K.exp(weights), axis=1)  + 1e-6))
		p_w = 0.5*K.sum(K.exp(pwm)*(pwm-weights)/(K.sum(K.exp(pwm), axis=1)  + 1e-6))
		return wt_kl*(1/(1+K.exp(-w_p)) + 1/(1+K.exp(-p_w)))
	return custom_reg

### Early Stopping
def custom_callback(file_checkpoint, quant_model='val_accuracy', monitor_metric='val_accuracy'):
    # monitor_metric: metric to be monitored
    # file_checkpoint: file path for saving the best model

    ret_callback = [
        EarlyStopping(monitor=monitor_metric, patience=1000, min_delta=0.0001, mode="auto"),
        ModelCheckpoint(filepath=file_checkpoint, monitor=quant_model, save_best_only=True, mode='auto'),
    ]
    return ret_callback

### Accuracy and Losses
def loss_mse(y_true, y_pred):
	lss_mse = []
	for i in range(y_true.shape[1]):
		lss_mse.append(np.mean(np.square(y_true[:, i]-y_pred[:, i])))
	lss_mse.append(np.mean(np.square(y_true-y_pred)))
	return lss_mse

def accuracy_binary_balanced(y_true, y_pred): 
	acc = []
	tp_t = 0
	for i in range(3):
		tp = np.sum(np.sum(y_true[:, i] == y_pred[:, i], axis=1) == 3)
		tp_t += tp
		acc_i = tp / (y_true.shape[0] + 1e-10)
		acc.append(round(acc_i, 3))
	acc_t = tp_t / (y_true.shape[0]*y_true.shape[1] + 1e-10)
	acc.append(round(acc_t, 3))
	return acc

def loss_mse_genewise(y_true, y_pred):
	return np.mean(np.square(y_true-y_pred), axis=1)


def accuracy_msk(model, msk, dict_features_labels):
	msk_seq = np.repeat(msk, 4).tolist()

	features_train = dict_features_labels['features_train'][:, :, msk_seq, :]
	arr_layer_train = dict_features_labels['arr_layer_train'][:, msk, :]
	labels_train = dict_features_labels['labels_train']
	features_valid = dict_features_labels['features_valid'][:, :, msk_seq, :]
	arr_layer_valid = dict_features_labels['arr_layer_valid'][:, msk, :]
	labels_valid = dict_features_labels['labels_valid']
	features_test = dict_features_labels['features_test'][:, :, msk_seq, :]
	arr_layer_test = dict_features_labels['arr_layer_test'][:, msk, :] 
	labels_test = dict_features_labels['labels_test']

	pred_train = model.predict([features_train, arr_layer_train])
	pred_valid = model.predict([features_valid, arr_layer_valid])
	pred_test = model.predict([features_test, arr_layer_test])

	y_true = np.concatenate([labels_train, labels_valid, labels_test])
	y_pred = np.concatenate([pred_train, pred_valid, pred_test])

	return loss_mse_genewise(y_true, y_pred)

def performance(dir_trial, cells):
	df_perf_summary = pd.DataFrame(
    columns=["Partition"]
    + ["var_" + cell for cell in cells]
    + ["var_total"]
    + ["mse_" + cell for cell in cells]
    + ["mse_total"]
	)
	file_perf = dir_trial + "performance.txt"
	df_perf = pd.read_csv(file_perf, sep="\t")
	i = 0
	for typ in ["train", "valid", "test"]:
	    row_var = [typ]
	    row_perf = []
	    file_pred_typ = dir_trial + "pred_" + typ + ".txt"
	    df_pred_typ = pd.read_csv(file_pred_typ, sep="\t")
	    for cell in cells:
	        row_var.append(np.var(df_pred_typ[cell]))
	        row_perf.append(df_perf[typ + "_mse_" + cell][0])
	    row_var.append(np.var(np.asarray(df_pred_typ[cells[1:]])))
	    row_perf.append(df_perf[typ + "_mse"][0])
	    df_perf_summary.loc[i] = row_var + row_perf
	    i += 1
	return df_perf_summary   

def accuracy(dir_trial, cells):
	df_perf = pd.DataFrame(columns=["Data"] + cells + ["Total"])

	df_pred = pd.DataFrame()
	for typ in ["train", "valid"]:
	    file_pred_typ = dir_trial + "pred_" + typ + ".txt"
	    df_pred_typ = pd.read_csv(file_pred_typ, sep="\t")
	    df_pred = pd.concat([df_pred, df_pred_typ], axis=0)
	    df_pred.index = range(len(df_pred))

	list_t_up_best = ["T_UP"]
	list_t_down_best = ["T_Down"]
	list_acc_best = ["Train + Valid"]
	for cell in cells:
	    acc_best = 0
	    t_up_best = 0
	    t_down_best = 0
	    for t_up in range(201):
	        for t_down in range(t_up + 1):
	            t_up_new = (t_up - 100) / 100
	            t_down_new = (t_down - 100) / 100
	            n_up_true = sum((df_pred[cell] == 1) & (df_pred["pred_" + cell] > t_up_new))
	            n_nc_true = sum(
	                (df_pred[cell] == 0)
	                & (
	                    (df_pred["pred_" + cell] <= t_up_new)
	                    & (df_pred["pred_" + cell] >= t_down_new)
	                )
	            )
	            n_down_true = sum(
	                (df_pred[cell] == -1) & (df_pred["pred_" + cell] < t_down_new)
	            )
	            acc = (n_up_true + n_nc_true + n_down_true) / len(df_pred)
	            if acc > acc_best:
	                acc_best = acc
	                t_up_best = t_up_new
	                t_down_best = t_down_new
	    list_t_up_best.append(t_up_best)
	    list_t_down_best.append(t_down_best)
	    list_acc_best.append(acc_best)
	    print(cell)

	list_acc_test = ["Test"]
	file_pred_test = dir_trial + "pred_test.txt"
	df_pred_test = pd.read_csv(file_pred_test, sep="\t")
	for i in range(1, 1+len(cells)):
	    cell = cells[i-1]
	    t_up = list_t_up_best[i]
	    t_down = list_t_down_best[i]
	    n_up_true = sum((df_pred_test[cell] == 1) & (df_pred_test["pred_" + cell] > t_up))
	    n_nc_true = sum(
	        (df_pred_test[cell] == 0)
	        & (
	            (df_pred_test["pred_" + cell] <= t_up)
	            & (df_pred_test["pred_" + cell] >= t_down)
	        )
	    )
	    n_down_true = sum(
	        (df_pred_test[cell] == -1) & (df_pred_test["pred_" + cell] < t_down)
	    )
	    acc_test = (n_up_true + n_nc_true + n_down_true) / len(df_pred_test)
	    list_acc_test.append(acc_test)

	df_perf.loc[0] = list_t_up_best + ["NA"]
	df_perf.loc[1] = list_t_down_best + ["NA"]
	df_perf.loc[2] = list_acc_best + [np.mean(list_acc_best[1:len(cells) + 1])]
	df_perf.loc[3] = list_acc_test + [np.mean(list_acc_test[1:len(cells) + 1])]

	return df_perf

#### Downstream Analysis
def pwm_ic(arr_pwm):
    log_pwm = np.log2((arr_pwm + 1e-100) / 0.25)
    return -round(np.sum(np.multiply(arr_pwm, log_pwm)), 3)

def kl_divergence_pwm(mat_a, mat_b):
    kl_d = 0.5 * np.sum(mat_a * (np.log(mat_a + 1e-10) - np.log(mat_b + 1e-10)))
    return kl_d

def ret_rc_pos(pos):
    pos = np.sum(np.multiply(np.array([0, 1, 2, 3]), pos))
    switcher = {
        0: [0, 0, 0, 1],
        1: [0, 0, 1, 0],
        2: [0, 1, 0, 0],
        3: [1, 0, 0, 0],
    }
    return switcher.get(pos)

def ret_rc_site(site):
    rc_site = []
    l_site = site.shape[1]
    for i in range(l_site):
        rc_site.append(ret_rc_pos(site[:, l_site - i - 1]))
    return np.array(rc_site).T

def ret_best_site(seq, fltr):
    best_score = -100
    l_pwm = fltr.shape[1]
    l_seq = seq.shape[2]
    for i in range(l_seq - l_pwm - 1):
        site_0 = seq[0, :, i : i + l_pwm]
        site_1 = seq[0, :, i : i + l_pwm]
        sc_0 = np.sum(np.multiply(fltr, site_0))
        sc_1 = np.sum(np.multiply(fltr, site_1))
        if sc_0 > sc_1:
            if sc_0 >= best_score:
                best_site = site_0
                best_score = sc_0
        else:
            if sc_1 >= best_score:
                best_site = ret_rc_site(site_1)
                best_score = sc_1
    return best_site, best_score