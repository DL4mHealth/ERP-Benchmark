export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training

# Disease Detection

# PD-SIM
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM --model Medformer --data MultiDatasets \
--training_datasets PD-SIM \
--testing_datasets PD-SIM \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD --model Medformer --data MultiDatasets \
--training_datasets PD-ODD \
--testing_datasets PD-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD --model Medformer --data MultiDatasets \
--training_datasets SCPD \
--testing_datasets SCPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD --model Medformer --data MultiDatasets \
--training_datasets RLPD \
--testing_datasets RLPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD --model Medformer --data MultiDatasets \
--training_datasets AOPD \
--testing_datasets AOPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI --model Medformer --data MultiDatasets \
--training_datasets ADHD-WMRI \
--testing_datasets ADHD-WMRI \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD --model Medformer --data MultiDatasets \
--training_datasets CESCA-AODD \
--testing_datasets CESCA-AODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD --model Medformer --data MultiDatasets \
--training_datasets CESCA-VODD \
--testing_datasets CESCA-VODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER --model Medformer --data MultiDatasets \
--training_datasets CESCA-FLANKER \
--testing_datasets CESCA-FLANKER \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# mTBI-ODD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-mTBI-ODD --model Medformer --data MultiDatasets \
--training_datasets mTBI-ODD \
--testing_datasets mTBI-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT --model Medformer --data MultiDatasets \
--training_datasets NSERP-MSIT \
--testing_datasets NSERP-MSIT \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method Medformer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD --model Medformer --data MultiDatasets \
--training_datasets NSERP-ODD \
--testing_datasets NSERP-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len_list 25,50,100 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15