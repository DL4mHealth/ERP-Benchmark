export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training

# Disease Detection

# PD-SIM
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-SIM --model ModernTCN --data MultiDatasets \
--training_datasets PD-SIM \
--testing_datasets PD-SIM \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# PD-ODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-PD-ODD --model ModernTCN --data MultiDatasets \
--training_datasets PD-ODD \
--testing_datasets PD-ODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# SCPD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-SCPD --model ModernTCN --data MultiDatasets \
--training_datasets SCPD \
--testing_datasets SCPD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# RLPD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-RLPD --model ModernTCN --data MultiDatasets \
--training_datasets RLPD \
--testing_datasets RLPD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# AOPD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-AOPD --model ModernTCN --data MultiDatasets \
--training_datasets AOPD \
--testing_datasets AOPD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# ADHD-WMRI
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-ADHD-WMRI --model ModernTCN --data MultiDatasets \
--training_datasets ADHD-WMRI \
--testing_datasets ADHD-WMRI \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --use_subject_vote --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


# ERP Task Classification
# CESCA-AODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD --model ModernTCN --data MultiDatasets \
--training_datasets CESCA-AODD \
--testing_datasets CESCA-AODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-VODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-VODD --model ModernTCN --data MultiDatasets \
--training_datasets CESCA-VODD \
--testing_datasets CESCA-VODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CESCA-FLANKER
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-CESCA-FLANKER --model ModernTCN --data MultiDatasets \
--training_datasets CESCA-FLANKER \
--testing_datasets CESCA-FLANKER \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# mTBI-ODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-mTBI-ODD --model ModernTCN --data MultiDatasets \
--training_datasets mTBI-ODD \
--testing_datasets mTBI-ODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-MSIT
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-MSIT --model ModernTCN --data MultiDatasets \
--training_datasets NSERP-MSIT \
--testing_datasets NSERP-MSIT \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# NSERP-ODD
python -u run.py --method ModernTCN \
--task_name supervised --is_training 1 --root_path ./dataset/200Hz/ --model_id S-NSERP-ODD --model ModernTCN --data MultiDatasets \
--training_datasets NSERP-ODD \
--testing_datasets NSERP-ODD \
--batch_size 512 --ffn_ratio 1 --patch_len 32 --stride 16 --num_blocks 1 1 1 --large_size 9 9 9 --small_size 5 5 5 --dims 32 64 128 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15