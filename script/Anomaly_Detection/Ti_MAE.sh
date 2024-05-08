torchrun \
    --standalone \
    --nproc_per_node=1 \
    run.py \
    --task_name anomaly_detection \
    --model Ti_MAE \
    --learning_rate 1e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 60 \
    --e_layers 2 \
    --d_model 256 \
    --activation "gelu" \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --dropout 0.3 \
    --linear_dropout 0.6 \
    --num_workers 10 \
    --use_scheduler \
    --input_len 12 \
    --weight_decay 1e-4 \
    --dataset TUAB \
    --root_path <root_path> \
    --log_dir <log_dir> \
    --pretrained_path <pretrained_path>
    