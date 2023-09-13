import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import time

from utils import accuracy_binary_balanced, custom_callback, custom_loss_func, loss_mse
from models import model_classifier
from output import write_pwms, plot_model_train_history

def trainer_classifier(dict_params, dict_features_labels):
	start = time.time()

	loss_func = dict_params['loss_func']
	x_train_pwm = dict_params['x_train_pwm']
	lrng_rt = dict_params['lrng_rt']
	train_pwm = dict_params['train_pwm']
	dir_checkpoint = dict_params['dir_logs']
	file_checkpoint = dir_checkpoint + 'trained_model.hdf'
	quant_monitor = dict_params['quant_monitor']
	quant_model = dict_params['quant_model']
	my_callback = custom_callback(file_checkpoint, quant_model, quant_monitor)
	cells = dict_params['cells']
	n_pwms = dict_params['n_pwms']
	n_epochs = dict_params['n_epochs']
	shape_lyrs = dict_params['shape_lyrs']

	features_train = dict_features_labels['features_train'] 
	arr_layer_train = dict_features_labels['arr_layer_train'] 
	labels_train = dict_features_labels['labels_train']
	features_valid = dict_features_labels['features_valid'] 
	arr_layer_valid = dict_features_labels['arr_layer_valid'] 
	labels_valid = dict_features_labels['labels_valid']
	features_test = dict_features_labels['features_test'] 
	arr_layer_test = dict_features_labels['arr_layer_test'] 
	labels_test = dict_features_labels['labels_test']

	model = model_classifier(dict_params)

	if dict_params['conv_init']:
		my_loss_func = custom_loss_func(loss_func)
	else:
		pwm_filters_train = dict_params['pwm_filters_train']
		print('Define Loss Func')
		print(pwm_filters_train.shape)
		my_loss_func = custom_loss_func(
		    loss_func,
		    model.layers[2].trainable_weights,
		    pwm_filters_train,
		    1,
		)
	print(my_loss_func)
	
	custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lrng_rt)
	
	model.compile(optimizer=custom_optimizer, loss=my_loss_func, metrics=["mean_squared_error"])
	history = model.fit(
			[features_train, arr_layer_train],
			labels_train,
			epochs=n_epochs,
			verbose=1,
			validation_data = ([features_valid, arr_layer_valid], labels_valid),
			callbacks=my_callback,
		)

	model_best = keras.models.load_model(file_checkpoint, custom_objects={"loss": my_loss_func})
	pred_train = model_best.predict([features_train, arr_layer_train])
	pred_valid = model_best.predict([features_valid, arr_layer_valid])
	pred_test = model_best.predict([features_test, arr_layer_test])
	print(pred_test[:2,:])
	loss_train = loss_mse(labels_train, pred_train)
	loss_valid = loss_mse(labels_valid, pred_valid)
	loss_test = loss_mse(labels_test, pred_test)
        
	Path(dir_checkpoint + 'training_plots/').mkdir(parents=False, exist_ok=True)
	file_plot = dir_checkpoint + 'training_plots/training_' + str(x_train_pwm) + '_' + train_pwm + '.pdf'
	plot_model_train_history(file_plot, history, loss_func)
	
	file_fltrs = dir_checkpoint + 'fltrs.wtmx'
	write_pwms(model_best.layers[2].trainable_weights[0], file_fltrs)

	file_fltr_wts = dir_checkpoint + 'fltr_wts.txt'
	model_fltr_wts = model_best.get_weights()[1]
	df_fltr_wts = pd.DataFrame()
	df_fltr_wts['PWMs'] = ['pwms_' + str(x) for x in range(n_pwms)]
	for i in range(len(cells)):
		df_fltr_wts[cells[i]] = model_fltr_wts[:, 0, 0, i]
	df_fltr_wts.loc[len(df_fltr_wts)] = ['Bias'] + list(model_best.get_weights()[2])
	df_fltr_wts.to_csv(file_fltr_wts, sep='\t', index=False, header=True)

	file_pred_wts = dir_checkpoint + 'pred_wts.wtmx'
	pred_wts = model_best.get_weights()[3+2*len(shape_lyrs)]
	pred_wts_bias = list(model_best.get_weights()[4+2*len(shape_lyrs)])
	df_pred_wts = pd.DataFrame()
	for i in range(pred_wts.shape[1]):
		df_pred_wts[str(i)] = list(pred_wts[:,i]) + [pred_wts_bias[i]]
	df_pred_wts.to_csv(file_pred_wts, sep='\t', index=False, header=True)

	end = time.time()
	val_ret = [x_train_pwm, train_pwm] + loss_train + loss_valid + loss_test + [round((end - start) / 60, 3)]
	print(mp.current_process(), val_ret)
	return val_ret
	# return 10

def pre_trainer_classifier(dict_params, dict_features_labels):
	start = time.time()

	loss_func = dict_params['loss_func']
	x_train_pwm = dict_params['x_train_pwm']
	lrng_rt = dict_params['lrng_rt']
	train_pwm = dict_params['train_pwm']
	dir_checkpoint = dict_params['dir_logs']
	file_checkpoint = dir_checkpoint + 'trained_model.hdf'
	quant_monitor = dict_params['quant_monitor']
	quant_model = dict_params['quant_model']
	my_callback = custom_callback(file_checkpoint, quant_model, quant_monitor)
	cells = dict_params['cells']
	n_pwms = dict_params['n_pwms']
	n_epochs = dict_params['n_epochs']

	features_train = dict_features_labels['features_train'] 
	arr_layer_train = dict_features_labels['arr_layer_train'] 
	labels_train = dict_features_labels['labels_train']
	features_valid = dict_features_labels['features_valid'] 
	arr_layer_valid = dict_features_labels['arr_layer_valid'] 
	labels_valid = dict_features_labels['labels_valid']
	features_test = dict_features_labels['features_test'] 
	arr_layer_test = dict_features_labels['arr_layer_test'] 
	labels_test = dict_features_labels['labels_test']

	model = dict_params['trained_model']

	if dict_params['conv_init']:
		my_loss_func = custom_loss_func(loss_func)
	else:
		pwm_filters_train = dict_params['pwm_filters_train']
		my_loss_func = custom_loss_func(
		    loss_func,
		    model.layers[2].trainable_weights[0],
		    pwm_filters_train,
		    1,
		)
	print(my_loss_func)
	
	custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lrng_rt)
	
	if dict_params['train_base_model']:
		model.compile(optimizer=custom_optimizer, loss=my_loss_func, metrics=["mean_squared_error"])
		history = model.fit(
				[features_train, arr_layer_train],
				labels_train,
				epochs=n_epochs,
				verbose=1,
				validation_data = ([features_valid, arr_layer_valid], labels_valid),
				callbacks=my_callback,
			)

		model_best = keras.models.load_model(file_checkpoint, custom_objects={"loss": my_loss_func})

		Path(dir_checkpoint + 'training_plots/').mkdir(parents=False, exist_ok=True)
		file_plot = dir_checkpoint + 'training_plots/training_' + str(x_train_pwm) + '_' + train_pwm + '.pdf'
		plot_model_train_history(file_plot, history, loss_func)

		file_fltrs = dir_checkpoint + 'fltrs_new.wtmx'
		write_pwms(model_best.layers[2].trainable_weights[0], file_fltrs)

		file_fltr_wts = dir_checkpoint + 'fltr_wts_new.txt'
		model_fltr_wts = model_best.get_weights()[1]
		df_fltr_wts = pd.DataFrame()
		df_fltr_wts['PWMs'] = ['pwms_' + str(x) for x in range(n_pwms)]
		for i in range(len(cells)):
			df_fltr_wts[cells[i]] = model_fltr_wts[:, 0, 0, i]
		df_fltr_wts.loc[len(df_fltr_wts)] = ['Bias'] + list(model_best.get_weights()[2])
		df_fltr_wts.to_csv(file_fltr_wts, sep='\t', index=False, header=True)

		file_pred_wts = dir_checkpoint + 'pred_wts_new.wtmx'
		pred_wts = model_best.get_weights()[5]
		pred_wts_bias = list(model_best.get_weights()[6])
		df_pred_wts = pd.DataFrame()
		for i in range(pred_wts.shape[1]):
			df_pred_wts[str(i)] = list(pred_wts[:,i]) + [pred_wts_bias[i]]
		df_pred_wts.to_csv(file_pred_wts, sep='\t', index=False, header=True)
	else:
		model_best = model

	pred_train = model_best.predict([features_train, arr_layer_train])
	pred_valid = model_best.predict([features_valid, arr_layer_valid])
	pred_test = model_best.predict([features_test, arr_layer_test])
	print(pred_test[:2,:])
	loss_train = loss_mse(labels_train, pred_train)
	loss_valid = loss_mse(labels_valid, pred_valid)
	loss_test = loss_mse(labels_test, pred_test)

	end = time.time()
	val_ret = [x_train_pwm, train_pwm] + loss_train + loss_valid + loss_test + [round((end - start) / 60, 3)]
	print(mp.current_process(), val_ret)
	return val_ret

def trainer_classifier_msk(dict_params, dict_features_labels):
	start = time.time()

	loss_func = dict_params['loss_func']
	x_train_pwm = dict_params['x_train_pwm']
	lrng_rt = dict_params['lrng_rt']
	train_pwm = dict_params['train_pwm']
	dir_checkpoint = dict_params['dir_logs']
	file_checkpoint = dir_checkpoint + 'trained_model.hdf'
	quant_monitor = dict_params['quant_monitor']
	quant_model = dict_params['quant_model']
	my_callback = custom_callback(file_checkpoint, quant_model, quant_monitor)
	cells = dict_params['cells']
	n_pwms = dict_params['n_pwms']
	n_epochs = dict_params['n_epochs']

	msk = dict_params['msk']
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

	model = model_classifier(dict_params)

	if dict_params['conv_init']:
		my_loss_func = custom_loss_func(loss_func)
	else:
		pwm_filters_train = dict_params['pwm_filters_train']
		my_loss_func = custom_loss_func(
		    loss_func,
		    model.layers[2].trainable_weights[0],
		    pwm_filters_train,
		    1,
		)
	print(my_loss_func)
	
	custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lrng_rt)
	
	model.compile(optimizer=custom_optimizer, loss=my_loss_func, metrics=["mean_squared_error"])
	history = model.fit(
			[features_train, arr_layer_train],
			labels_train,
			epochs=n_epochs,
			verbose=1,
			validation_data = ([features_valid, arr_layer_valid], labels_valid),
			callbacks=my_callback,
		)

	model_best = keras.models.load_model(file_checkpoint, custom_objects={"loss": my_loss_func})
	pred_train = model_best.predict([features_train, arr_layer_train])
	pred_valid = model_best.predict([features_valid, arr_layer_valid])
	pred_test = model_best.predict([features_test, arr_layer_test])
	print(pred_test[:2,:])
	loss_train = loss_mse(labels_train, pred_train)
	loss_valid = loss_mse(labels_valid, pred_valid)
	loss_test = loss_mse(labels_test, pred_test)
        
	Path(dir_checkpoint + 'training_plots/').mkdir(parents=False, exist_ok=True)
	file_plot = dir_checkpoint + 'training_plots/training_' + str(x_train_pwm) + '_' + train_pwm + '.pdf'
	plot_model_train_history(file_plot, history, loss_func)
	
	if dict_params['conv_init']:
		file_fltrs = dir_checkpoint + 'fltrs.wtmx'
		write_pwms(model_best.layers[2].trainable_weights[0], file_fltrs)

		file_fltr_wts = dir_checkpoint + 'fltr_wts.txt'
		model_fltr_wts = model_best.get_weights()[1]
		df_fltr_wts = pd.DataFrame()
		df_fltr_wts['PWMs'] = ['pwms_' + str(x) for x in range(n_pwms)]
		for i in range(len(cells)):
			df_fltr_wts[cells[i]] = model_fltr_wts[:, 0, 0, i]
		df_fltr_wts.loc[len(df_fltr_wts)] = ['Bias'] + list(model_best.get_weights()[2])
		df_fltr_wts.to_csv(file_fltr_wts, sep='\t', index=False, header=True)

		file_pred_wts = dir_checkpoint + 'pred_wts.wtmx'
		pred_wts = model_best.get_weights()[5]
		pred_wts_bias = list(model_best.get_weights()[6])
		df_pred_wts = pd.DataFrame()
		for i in range(pred_wts.shape[1]):
			df_pred_wts[str(i)] = list(pred_wts[:,i]) + [pred_wts_bias[i]]
		df_pred_wts.to_csv(file_pred_wts, sep='\t', index=False, header=True)

	end = time.time()
	val_ret = [x_train_pwm, train_pwm] + loss_train + loss_valid + loss_test + [round((end - start) / 60, 3)]
	print(mp.current_process(), val_ret)
	return val_ret, model_best