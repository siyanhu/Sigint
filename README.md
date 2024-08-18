# Sigint
1. Data Preparation

   download all folders from the following link

   https://hkustconnect-my.sharepoint.com/personal/shuas_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fshuas%5Fconnect%5Fust%5Fhk%2FDocuments%2F%5BProject%5D%5BARKit%20Survey%5D%2Fsurvey%5Fdata%2FHkust%5F20240814&noAuthRedirect=1

   
   store them under root_dir/path_num/, for example hkust4f/path1

   run `python preprocess.py`

   


2. Train & Test

   files under src are not modified
   
   add scheduler and early stopping in `lstm_train_sigint.py`

   modify directory names, input and output size
   
   example:

   `python lstm_train_sigint.py --sequence_length 200 --hidden_sizes 64 32 16 --num_epochs 500 --model_type lstm`
   
