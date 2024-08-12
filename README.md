# Sigint
1. Data Preparation

   download all folders starting with HKUST_fusion_4F

   store them under root_dir/path_num/

   run `python preprocess.py`


2. Train & Test

   files under src are not modified
   
   add scheduler and early stopping in `lstm_train_sigint.py`

   modify directory names, input and output size
   
   example:

   `python lstm_train_sigint.py --sequence_length 300 --hidden_sizes 128 64 32 16 --num_epochs 500`
   
