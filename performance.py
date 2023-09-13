import pandas as pd
from pathlib import Path
import sys

from utils import accuracy, performance 

dict_cmnd_input = {}
for i in range(len(sys.argv)):
	if sys.argv[i] == '-dir_trial':
			dict_cmnd_input['dir_trial'] = sys.argv[i+1]

dir_trial = dict_cmnd_input['dir_trial']
if dir_trial[-1] != '/':
	dir_trial = dir_trial + '/'			

dir_analysis = dir_trial + "analysis/"
Path(dir_analysis).mkdir(parents=False, exist_ok=True)

file_perf = dir_trial + "performance.txt"
df_perf = pd.read_csv(file_perf, sep="\t")
n_cells = int((len(df_perf.columns) - 3)/4)
cells = [x[10:] for x in df_perf.columns[2:2+n_cells]]
print(dir_trial, cells)

#### Variance and MSE
file_perf_summary = dir_analysis + "performance_summary.txt"
df_perf_summary = performance(dir_trial, cells)
df_perf_summary.to_csv(file_perf_summary, sep="\t", index=False, header=True)
# print(df_perf_summary)

#### Accuracy
file_accuracy = dir_analysis + "accuracy.txt"
df_accuracy = accuracy(dir_trial, cells)
df_accuracy.to_csv(file_accuracy, sep="\t", index=False, header=True)
# print(df_accuracy)