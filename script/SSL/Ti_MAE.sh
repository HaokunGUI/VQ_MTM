torchrun \
    --standalone \
    --nproc_per_node=1 \
    run.py \
    --task_name ssl \
    --model Ti_MAE \
    --learning_rate 2e-4 \
    --normalize \
    --patience 0 \
    --num_epochs 100 \
    --e_layers 2 \
    --d_layers 1 \
    --d_model 256 \
    --activation "gelu" \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --dropout 0.3 \
    --linear_dropout 0.2 \
    --mask_ratio 0.3 \
    --num_workers 1 \
    --use_scheduler \
    --dataset TUAB \
    --root_path <root_path> \
    --log_dir <log_dir> \
    --warmup_epochs 20