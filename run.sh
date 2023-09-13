declare trial=trial_sample
declare dir_logs=$trial/logs
mkdir $dir_logs
python 1_classifier.py -config $trial/config.txt > $dir_logs/logs.txt 
python performance.py -dir_trial $trial > $dir_logs/logs_perf.txt 
python 2_classifier_msk_trnd.py -config $trial/config_nn4.txt > $dir_logs/logs_msk_trnd.txt 
python 3_msks_best_search.py -config $trial/config_nn4.txt > $dir_logs/logs_msk_best_search.txt 
python 4_classifier_msk_best.py -config $trial/config_nn4.txt > $dir_logs/logs_msk_best.txt 
python performance.py -dir_trial $trial/msk_best > $dir_logs/logs_perf_best.txt 
python 5_classifier_msk_best_kdtf.py -config $trial/config_nn4.txt > $dir_logs/logs_msk_best_kdtf.txt 
python 6_pwms.py -dir_trial $trial > $dir_logs/logs_pwms.txt