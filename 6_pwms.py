import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pathlib import Path
import os
import scipy.stats as ss
import shutil
import sys
import time

from shutil import copy

from utils import nucleotide2ind, ret_rc_pos, ret_rc_site, ret_best_site, pwm_ic, kl_divergence_pwm
from data_loader import load_config, load_exp


dict_cmnd_input = {}
for i in range(len(sys.argv)):
    if sys.argv[i] == '-dir_trial':
        dict_cmnd_input['dir_trial'] = sys.argv[i+1]
        dir_trial = dict_cmnd_input['dir_trial']
        if dir_trial[-1] != '/':
            dir_trial = dir_trial + '/' 
        file_config = dir_trial + 'config_nn4.txt'
        file_fltrs = dir_trial + 'msk_best/fltrs_new.wtmx'
    elif sys.argv[i] == '-file_fltrs':
        file_fltrs = sys.argv[i+1]
    elif sys.argv[i] == '-file_config':
        file_config = sys.argv[i+1]   	

dir_analysis = dir_trial + "analysis/"
dict_config = load_config(file_config) 
cells = dict_config['cells']
dir_seqs = dict_config["dir_data"] + dict_config["dir_seqs"]
dir_prtn = dict_config["dir_data"] + dict_config["dir_prtn"]
file_exp = dir_prtn + dict_config["file_gene_exp"]

#### Consolidate all best enhancers into a single file for every cell type
Path(dir_analysis + "msks/best_enhancers/").mkdir(parents=False, exist_ok=True)

file_be = dir_analysis + "msks/best_enhancers.txt"
df_be = pd.read_csv(file_be, sep="\t")

df_exp = load_exp(file_exp)

dict_enhancers = {}
for gene in df_exp[df_exp.columns[0]]:
    file_seq_gene = dir_seqs + gene + '.fa'
    for line in open(file_seq_gene):
        trans = line.strip()
        if trans[0] == '>':
            enhancer = trans[1:]
        else:
            dict_enhancers[enhancer] = trans


for cell in cells:
    list_be_cell = list(df_be[cell].unique())

    file_be_cell = dir_analysis + "msks/best_enhancers/" + cell + ".fa"
    f = open(file_be_cell, "w")
    for be_cell in list_be_cell:
        f.write(">" + be_cell + "\n" + dict_enhancers[be_cell] + "\n")
    f.close()
    print(cell)

##### Generate PWMs
dir_pwms_be_bs = dir_trial + 'msk_best/pwms_be_bs/'
Path(dir_pwms_be_bs).mkdir(parents=False, exist_ok=True)

dict_fltrs = {}
for line in open(file_fltrs):
    trans = line.strip().split()
    if trans[0][0] == ">":
        fltr = trans[0][1:]
        dict_fltrs[fltr] = []
    elif trans[0][0] == "<":
        dict_fltrs[fltr] = np.array(dict_fltrs[fltr]).T
    else:
        dict_fltrs[fltr].append([float(x) for x in trans])

dict_fltrs_be = {}
for fltr in dict_fltrs:
    dict_fltrs_be[fltr] = np.zeros(dict_fltrs[fltr].shape)

for cell in cells: 
    time_start = time.time()
    file_be_cell = dir_analysis + "msks/best_enhancers/" + cell + ".fa"
    for line in open(file_be_cell):
        seq = line.strip()
        if seq[0] != ">":
            l_seq = len(seq)
            seq_oh = np.zeros((2, 4, l_seq))
            for i in range(l_seq):
                ntnum = nucleotide2ind(seq[i])
                seq_oh[0, ntnum, np.array([i])] = 1
                seq_oh[1, ntnum, l_seq - 1 - np.array([i])] = 1
            for fltr in dict_fltrs_be.keys():
                best_site, best_score = ret_best_site(seq_oh, dict_fltrs[fltr])
                if best_score > 0:
                    dict_fltrs_be[fltr] = dict_fltrs_be[fltr] + best_site
    time_end = time.time()
    print(cell, round((time_end - time_start) / 60, 3))

dict_pwms_be = {}
for fltr in dict_fltrs_be.keys():
    pwm = ">pwms_" + fltr.split("_")[1]
    dict_pwms_be[pwm] = (dict_fltrs_be[fltr] / np.sum(dict_fltrs_be[fltr], axis=0)).T


file_pwms_be = dir_pwms_be_bs + "pwms_be_bs.wtmx"
f = open(file_pwms_be, "w")
for pwm in dict_pwms_be:
    arr_pwm = dict_pwms_be[pwm]
    f.write(pwm + "\t" + str(arr_pwm.shape[0]) + "\n")
    for i in range(arr_pwm.shape[0]):
        line = ""
        for j in range(4):
            line = line + str(arr_pwm[i, j]) + "\t"
        f.write(line + "\n")
    f.write("<" + "\n")
f.close()

file_pwms_tomtom = dir_pwms_be_bs + "pwms_be_bs_tomtom.txt"
f = open(file_pwms_tomtom, "w")
f.write("MEME version 5.5.0" + "\n" + "\n")
f.write("ALPHABET= ACGT" + "\n" + "\n")
f.write("strands: + -" + "\n" + "\n")
f.write(
    "Background letter frequencies" + "\n" + "A 0.25 C 0.25 G 0.25 T 0.25" + "\n" + "\n"
)
for pwm in dict_pwms_be.keys():
    arr_pwm = dict_pwms_be[pwm]
    f.write("MOTIF" + "\t" + pwm[1:] + "\n")
    f.write(
        "letter-probability matrix: " + "alength= 4 w= " + str(arr_pwm.shape[0]) + "\n"
    )
    for i in range(arr_pwm.shape[0]):
        list_probs = list(arr_pwm[i, :])
        f.write(
            str(list_probs[0])
            + "\t"
            + str(list_probs[1])
            + "\t"
            + str(list_probs[2])
            + "\t"
            + str(list_probs[3])
            + "\n"
        )
    f.write("\n")
f.close()

file_pwm_info = dir_pwms_be_bs + "pwms_be_bs_info.txt"
df_pwm_info = pd.DataFrame(columns=["PWMs", "IC"])
i_pwm = 0
for pwm in dict_pwms_be.keys():
    df_pwm_info.loc[i_pwm] = [pwm[1:], pwm_ic(dict_pwms_be[pwm])]
    i_pwm += 1
df_pwm_info.to_csv(file_pwm_info, sep="\t", index=False, header=True)

file_ic_plot = dir_pwms_be_bs + "pwms_be_bs_ic.pdf"
fig = plt.figure(figsize=(5, 4))
plt.hist(df_pwm_info["IC"], bins=20)
plt.savefig(file_ic_plot, dpi=fig.dpi)
plt.close()

file_kl_d_plot = dir_pwms_be_bs + "pwms_be_bs_inter_kl_d.pdf"
file_kl_d = dir_pwms_be_bs + "pwms_be_bs_inter_kl_d.txt"
df_inter_pwm_kl_d = pd.DataFrame()
df_inter_pwm_kl_d["PWMs"] = [x[1:] for x in dict_pwms_be.keys()]
for pwm1 in df_inter_pwm_kl_d["PWMs"]:
    list_kl_d_pwms = []
    arr_pwm1 = dict_pwms_be[">" + pwm1]
    for pwm2 in df_inter_pwm_kl_d["PWMs"]:
        arr_pwm2 = dict_pwms_be[">" + pwm2]
        kl_d_total = kl_divergence_pwm(arr_pwm1, arr_pwm2) + kl_divergence_pwm(
            arr_pwm2, arr_pwm1
        )
        list_kl_d_pwms.append(kl_d_total)
    df_inter_pwm_kl_d[pwm1] = list_kl_d_pwms
df_inter_pwm_kl_d.index = df_inter_pwm_kl_d["PWMs"]
df_inter_pwm_kl_d.to_csv(file_kl_d, sep="\t", index=False, header=True)