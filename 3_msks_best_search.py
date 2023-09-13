import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pathlib import Path
import os
import scipy.stats as ss
#import seaborn as sns
import shutil
import time

import gc
import json
import multiprocessing as mp
from multiprocessing import Pool
import psutil
import sys

from shutil import copy

from data_loader import load_acc_msk, load_config, load_exp, load_prtn, load_seqs_msk, load_base_model, load_msk, load_pred
from data_creator import create_dir, create_exp_labels, create_features_labels, create_model_params, create_pwm_filters, create_pwm_init
from models import model_classifier, load_trained_model
from trainer import pre_trainer_classifier, trainer_classifier
from utils import create_enhs_configs, create_masks, accuracy_msk
from output import write_msks
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

cells_msks = []
for cell in cells:
    cells_msks = cells_msks + [cell + "_" + str(x) for x in range(1, 5)]

dir_analysis = dir_logs + "analysis/"
dir_output = dir_analysis + "msks/"
Path(dir_analysis).mkdir(parents=False, exist_ok=True)
Path(dir_output).mkdir(parents=False, exist_ok=True)

df_genes = load_exp(file_exp)

file_msks = dir_logs + "enhancers_masks.txt"
dict_msks = {}
for i, line in enumerate(open(file_msks)):
    if i > 0:
        trans = line.strip().split()
        dict_msks[trans[0]] = np.array([bool(int(x)) for x in trans[1:]])

dict_msks_allowed = {}
for gene in df_genes["Genes"]:
    dict_msks_allowed[gene] = []
    file_acc_gene = dir_acc + gene + ".txt"
    list_peaks_genes = []
    for i, line in enumerate(open(file_acc_gene)):
        if i % 2 == 0:
            trans = line.strip()
            list_peaks_genes.append(trans[1:])
    arr_peaks_genes = np.array(list_peaks_genes)
    for msk in dict_msks.keys():
        if sum([int("Dummy" in x) for x in arr_peaks_genes[dict_msks[msk]]]) == 0:
            dict_msks_allowed[gene].append(msk)

dict_msk_pred = {}
for gene in df_genes["Genes"]:
    dict_msk_pred[gene] = {}
    for cell in cells[1:]:
        dict_msk_pred[gene][cell] = []

for i in range(256):
    df_pred_msk = load_pred(dir_logs + "msk_pred/" + str(i) + "/")
    for j in range(len(df_pred_msk["Genes"])):
        gene = df_pred_msk["Genes"][j]
        if "msk_" + str(i) in dict_msks_allowed[gene]:
            for cell in cells[1:]:
                dict_msk_pred[gene][cell].append(df_pred_msk["pred_" + cell][j])

dict_rel_msks = {}
dict_improv_msks = {}
for i in range(16):
    file_perf = dir_logs + "msk/enhancer_performance_" + str(i) + ".txt"
    if i == 0:
        for j, line in enumerate(open(file_perf)):
            trans = line.strip().split()
            if j == 0:
                msks = trans
            else:
                gene = int(trans[0])
                if gene in range(len(df_genes)):
                    gene_msk0 = float(trans[1])
                    dict_rel_msks[gene] = {}
                    dict_improv_msks[gene] = [0] * 16
                    dict_rel_msks[gene]["msk_0"] = gene_msk0
                    for k in range(1, len(trans)):
                        if msks[k] in dict_msks_allowed[df_genes["Genes"][gene]]:
                            if float(trans[k]) < gene_msk0:
                                dict_rel_msks[gene][msks[k]] = float(trans[k])
                                dict_improv_msks[gene] = np.add(
                                    dict_improv_msks[gene], dict_msks[msks[k]]
                                )
    else:
        for j, line in enumerate(open(file_perf)):
            trans = line.strip().split()
            if j == 0:
                msks = trans
            else:
                gene = int(trans[0])
                if gene in range(len(df_genes)):
                    gene_msk0 = dict_rel_msks[gene]["msk_0"]
                    for k in range(1, len(trans)):
                        if msks[k] in dict_msks_allowed[df_genes["Genes"][gene]]:
                            if float(trans[k]) < gene_msk0:
                                dict_rel_msks[gene][msks[k]] = float(trans[k])
                                dict_improv_msks[gene] = np.add(
                                    dict_improv_msks[gene], dict_msks[msks[k]]
                                )

file_improv_perf = dir_output + "enhancer_relevant.txt"
df_improv_perf = pd.DataFrame(columns=["Genes", "Idx", "nImprov"] + cells_msks)
i = 0
count_miss = 0
for gene in range(len(df_genes)):
    if gene in dict_improv_msks:
        # for gene in dict_improv_msks.keys():
        df_improv_perf.loc[i] = [
            df_genes["Genes"][gene],
            gene,
            np.sum(dict_improv_msks[gene]),
        ] + list(dict_improv_msks[gene])
    else:
        df_improv_perf.loc[i] = [df_genes["Genes"][gene], gene, 0,] + [0] * 16
        count_miss += 1
    i += 1
df_improv_perf.to_csv(file_improv_perf, sep="\t", index=False, header=True)


file_best_msks = dir_output + "best_msks.txt"
dict_best_msks = {}
for i in range(16):
    file_perf = dir_logs + "msk/enhancer_performance_" + str(i) + ".txt"
    if i == 0:
        for j, line in enumerate(open(file_perf)):
            trans = line.strip().split()
            if j == 0:
                msks = trans
            else:
                gene = int(trans[0])
                dict_best_msks[gene] = {}
                gene_mask_best_mse = 1e10
                for k in range(1, len(trans)):
                    if msks[k] in dict_msks_allowed[df_genes["Genes"][gene]]:
                        if float(trans[k]) < gene_mask_best_mse:
                            gene_mask_best_mse = float(trans[k])
                            dict_best_msks[gene]["Genes"] = df_genes["Genes"][gene]
                            dict_best_msks[gene]["GeneID"] = gene
                            dict_best_msks[gene]["Msk"] = msks[k]
                            dict_best_msks[gene]["MSE"] = gene_mask_best_mse
    else:
        for j, line in enumerate(open(file_perf)):
            trans = line.strip().split()
            if j == 0:
                msks = trans
            else:
                gene = int(trans[0])
                gene_mask_best_mse = dict_best_msks[gene]["MSE"]
                for k in range(1, len(trans)):
                    if msks[k] in dict_msks_allowed[df_genes["Genes"][gene]]:
                        if float(trans[k]) < gene_mask_best_mse:
                            gene_mask_best_mse = float(trans[k])
                            dict_best_msks[gene]["Genes"] = df_genes["Genes"][gene]
                            dict_best_msks[gene]["GeneID"] = gene
                            dict_best_msks[gene]["Msk"] = msks[k]
                            dict_best_msks[gene]["MSE"] = gene_mask_best_mse
df_best_msks = pd.DataFrame.from_dict(dict_best_msks, orient="index")
df_best_msks.index = range(len(df_best_msks))
df_best_msks.to_csv(file_best_msks, sep="\t", index=False, header=True)


file_best_enhancer = dir_output + "best_enhancers.txt"
df_best_enhancer = pd.DataFrame(columns=["Genes"] + cells)
for i_gene in range(len(df_best_msks)):
    gene = df_best_msks["Genes"][i_gene]
    best_msk = np.array(dict_msks[df_best_msks["Msk"][i_gene]])
    file_acc_gene = dir_acc + gene + ".txt"
    list_peaks_genes = []
    for i, line in enumerate(open(file_acc_gene)):
        if i % 2 == 0:
            trans = line.strip()
            list_peaks_genes.append(trans[1:])
    arr_peaks_genes = np.array(list_peaks_genes)
    df_best_enhancer.loc[i_gene] = [gene] + list(arr_peaks_genes[best_msk])
df_best_enhancer.to_csv(file_best_enhancer, sep="\t", index=False, header=True)

Path(dir_logs + "gene_msks").mkdir(parents=False, exist_ok=True)
for gene in df_genes["Genes"]:
    if gene in df_best_msks["Genes"].values:
        msk = df_best_msks["Msk"][df_best_msks["Genes"] == gene].values[0]
    else:
        msk = "msk_0"
    gene_msk = [int(x) for x in dict_msks[msk]]
    f = open(dir_logs + "gene_msks/" + gene + ".txt", "w")
    for i in range(len(gene_msk)):
        f.write(">enh_" + str(i) + "\n" + str(gene_msk[i]) + "\n")
    f.close()