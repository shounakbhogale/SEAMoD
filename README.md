# SEAMoD

Required Python Pacakges - 
  gc
  json
  matplotlib
  multiprocessing
  numpy
  os
  pandas
  pathlib
  psutil
  scipy
  shutil
  sys
  tensorflow
  time

Sample input is provided in the directory named 'data'.
Running the bash script 'run.sh' will - 
  1. Train the model using using nearest enhancers 
  2. Build a library of optimal enhancers using the model learned in the previous step
  3. Retrian the model using optimal enhancers
  4. Generate a list of important motifs learned by the model in the previous step.

A sample output (using the sample input) is provided in the directory 'trial_sample'.
