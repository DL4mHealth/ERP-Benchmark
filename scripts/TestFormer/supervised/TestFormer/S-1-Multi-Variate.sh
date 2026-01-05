export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training

# Disease Detection

# PD-SIM
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets PD-SIM \
--testing_datasets PD-SIM \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets PD-ODD \
--testing_datasets PD-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets SCPD \
--testing_datasets SCPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets RLPD \
--testing_datasets RLPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets AOPD \
--testing_datasets AOPD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets ADHD-WMRI \
--testing_datasets ADHD-WMRI \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets CESCA-AODD \
--testing_datasets CESCA-AODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets CESCA-VODD \
--testing_datasets CESCA-VODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets CESCA-FLANKER \
--testing_datasets CESCA-FLANKER \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# mTBI-ODD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-mTBI-ODD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets mTBI-ODD \
--testing_datasets mTBI-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets NSERP-MSIT \
--testing_datasets NSERP-MSIT \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method TestFormer \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD-Multi-Variate --model TestFormer --data MultiDatasets \
--training_datasets NSERP-ODD \
--testing_datasets NSERP-ODD \
--e_layers 6 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --patch_len 25 --patch_type multi-variate --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15