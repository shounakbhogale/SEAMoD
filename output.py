import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

### Write first convolutional layer filters into a file.
def write_pwms(fltrs, file_fltrs):
	f = open(file_fltrs, 'w')
	for i in range(fltrs.shape[4]):
		f.write('>fltrs_' + str(i))
		for j in range(fltrs.shape[2]):
			f.write('\n'+str(float(fltrs[0,0,j,0,i]))+'\t'+str(float(fltrs[0,1,j,0,i]))+'\t'+str(float(fltrs[0,2,j,0,i]))+'\t'+str(float(fltrs[0,3,j,0,i])))
		f.write('\n'+'<'+'\n')

### Write second layer (FC) weights
def write_fltr_wts(model, cells, file_fltr_wts):
	model_fltr_wts = model.get_weights()[1]
	df_fltr_wts = pd.DataFrame()
	df_fltr_wts['PWMs'] = ['pwms_' + str(x) for x in range(model_fltr_wts.shape[0])]
	for i in range(len(cells)):
		df_fltr_wts[cells[i]] = model_fltr_wts[:, 0, 0, i]
	df_fltr_wts.loc[len(df_fltr_wts)] = ['Bias'] + list(model.get_weights()[2])
	df_fltr_wts.to_csv(file_fltr_wts, sep='\t', index=False, header=True)

### Write LR scores
def write_gene_lr(file_gene_lr, arr_gene_lr):
	df_gene_lr = pd.DataFrame(arr_gene_lr)
	df_gene_lr.columns  = ['pwms_' + str(x) for x in range(arr_gene_lr.shape[1])]
	df_gene_lr.to_csv(file_gene_lr, sep='\t', index=False, header=True) 


def write_lr(arr_lr, dir_lr, lis_genes, n_cpu):
	start = time.time()
	n_genes = arr_lr.shape[0]
	n_process = int(np.ceil(n_genes/n_cpu))
	count_lr = []
	for i in range(n_process):
		pool_process = Pool(n_cpu)
		count_lr = count_lr + pool_process.starmap(write_gene_lr, [(dir_lr + lis_genes[x] + '.txt', arr_lr[x,:,:]) for x in range(i*n_cpu, min((i+1)*n_cpu, n_genes))])
		pool_process.close()
		pool_process.join()
	count_lr = np.asarray(count_lr)
	print('LR scores done. ', count_lr.shape, round((time.time() - start)/60, 3))

### Plot training history
def plot_model_train_history(file_plot, history, loss_func):
	fig = plt.figure(figsize=(20, 10))
	fig.tight_layout()
	plt.subplot(1, 2, 1)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss: ' + str(loss_func))
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.subplot(1, 2, 2)
	plt.plot(history.history['mean_squared_error'])
	plt.plot(history.history['val_mean_squared_error'])
	plt.title('MSE')
	plt.ylabel('MSE')
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	fig.savefig(file_plot, dpi=fig.dpi)
	plt.close()

### Write enhancer masks
def write_msks(list_msks, file_msks):
	df_msks = pd.DataFrame(columns=['Msks'] + ['enh_' + str(x) for x in range(len(list_msks[0]))])
	for i in range(len(list_msks)):
		df_msks.loc[i] = ['msk_' + str(i)] + [int(x) for x in list_msks[i]]
	df_msks.to_csv(file_msks, sep='\t', index=False, header=True) 