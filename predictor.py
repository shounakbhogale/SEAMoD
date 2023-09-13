import gc
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from tensorflow import keras

from models import model_classifier
from utils import custom_loss_func

def predictor_msk(dict_params, dict_features_labels, dict_prtn, msk, df_exp, base_model=None):
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

	rows_train = dict_prtn['train']
	rows_valid = dict_prtn['valid']
	rows_test = dict_prtn['test']

	dir_output = dict_params['dir_output']

	cells = dict_params['cells']
	col_names = ['pred_' + cell for cell in cells[1:]]

	file_pred_train = dir_output + '/pred_train.txt'
	pred_train = base_model.predict([features_train, arr_layer_train])
	df_pred_train = pd.DataFrame(pred_train, columns=col_names)
	df_gt_train = df_exp.iloc[rows_train]
	df_gt_train.index = range(len(df_gt_train))
	df_pred_train_output = pd.concat([df_gt_train, df_pred_train], axis=1)
	df_pred_train_output.to_csv(file_pred_train, sep='\t', index=False, header=True) 

	file_pred_valid = dir_output + '/pred_valid.txt'
	pred_valid = base_model.predict([features_valid, arr_layer_valid])
	df_pred_valid = pd.DataFrame(pred_valid, columns=col_names)
	df_gt_valid = df_exp.iloc[rows_valid]
	df_gt_valid.index = range(len(df_gt_valid))
	df_pred_valid_output = pd.concat([df_gt_valid, df_pred_valid], axis=1)
	df_pred_valid_output.to_csv(file_pred_valid, sep='\t', index=False, header=True)

	file_pred_test = dir_output + '/pred_test.txt'
	pred_test = base_model.predict([features_test, arr_layer_test])
	df_pred_test = pd.DataFrame(pred_test, columns=col_names)
	df_gt_test = df_exp.iloc[rows_test]
	df_gt_test.index = range(len(df_gt_test))
	df_pred_test_output = pd.concat([df_gt_test, df_pred_test], axis=1)
	df_pred_test_output.to_csv(file_pred_test, sep='\t', index=False, header=True)

	del df_pred_train_output, df_pred_train, df_gt_train
	del df_pred_valid_output, df_pred_valid, df_gt_valid
	del df_pred_test_output, df_pred_test, df_gt_test
	gc.collect()
	print('Deleted dict_data. Memory Usage: ' + str(psutil.cpu_percent()))

def predictor_raw(dict_params, dict_features_labels, dict_prtn, df_exp, base_model=None):	
	features_train = dict_features_labels['features_train'] 
	arr_layer_train = dict_features_labels['arr_layer_train'] 
	labels_train = dict_features_labels['labels_train']
	features_valid = dict_features_labels['features_valid'] 
	arr_layer_valid = dict_features_labels['arr_layer_valid'] 
	labels_valid = dict_features_labels['labels_valid']
	features_test = dict_features_labels['features_test'] 
	arr_layer_test = dict_features_labels['arr_layer_test'] 
	labels_test = dict_features_labels['labels_test']

	rows_train = dict_prtn['train']
	rows_valid = dict_prtn['valid']
	rows_test = dict_prtn['test']

	cells = dict_params['cells']
	col_names = ['pred_' + cell for cell in cells[1:]]

	dir_output = dict_params['dir_output']

	if base_model:
		model_best = base_model
	else:
		loss_func = dict_params['loss_func']
		x_train_pwm = dict_params['x_train_pwm']
		train_pwm = dict_params['train_pwm']
		dir_checkpoint = dict_params['dir_logs']
		file_checkpoint = dir_checkpoint + 'trained_model.hdf'
		model = model_classifier(dict_params)
		if dict_params['conv_init']:
			my_loss_func = custom_loss_func(loss_func)
		else:
			pwm_filters_train = dict_params['pwm_filters_train']
			my_loss_func = custom_loss_func(
			    loss_func,
			    model.layers[3].trainable_weights[0],
			    pwm_filters_train,
			    1,
			)
		model_best = keras.models.load_model(file_checkpoint, custom_objects={"loss": my_loss_func})

	file_pred_train = dir_output + '/pred_train.txt'
	pred_train = model_best.predict([features_train, arr_layer_train])
	df_pred_train = pd.DataFrame(pred_train, columns=col_names)
	df_gt_train = df_exp.iloc[rows_train]
	df_gt_train.index = range(len(df_gt_train))
	df_pred_train_output = pd.concat([df_gt_train, df_pred_train], axis=1)
	df_pred_train_output.to_csv(file_pred_train, sep='\t', index=False, header=True) 

	file_pred_valid = dir_output + '/pred_valid.txt'
	pred_valid = model_best.predict([features_valid, arr_layer_valid])
	df_pred_valid = pd.DataFrame(pred_valid, columns=col_names)
	df_gt_valid = df_exp.iloc[rows_valid]
	df_gt_valid.index = range(len(df_gt_valid))
	df_pred_valid_output = pd.concat([df_gt_valid, df_pred_valid], axis=1)
	df_pred_valid_output.to_csv(file_pred_valid, sep='\t', index=False, header=True)

	file_pred_test = dir_output + '/pred_test.txt'
	pred_test = model_best.predict([features_test, arr_layer_test])
	df_pred_test = pd.DataFrame(pred_test, columns=col_names)
	df_gt_test = df_exp.iloc[rows_test]
	df_gt_test.index = range(len(df_gt_test))
	df_pred_test_output = pd.concat([df_gt_test, df_pred_test], axis=1)
	df_pred_test_output.to_csv(file_pred_test, sep='\t', index=False, header=True)

	del df_pred_train_output, df_pred_train, df_gt_train
	del df_pred_valid_output, df_pred_valid, df_gt_valid
	del df_pred_test_output, df_pred_test, df_gt_test
	gc.collect()
	print('Deleted dict_data. Memory Usage: ' + str(psutil.cpu_percent()))