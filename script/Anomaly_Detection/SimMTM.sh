torchrun \
    --standalone \
    --nproc_per_node=1 \
    run.py \
    --task_name anomaly_detection \
    --model SimMTM \
    --learning_rate 1e-3 \
    --normalize \
    --patience 0 \
    --num_epochs 60 \
    --e_layers 2 \
    --d_model 256 \
    --activation "gelu" \
    --linear_dropout 0.6 \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --input_len 12 \
    --output_len 60 \
    --mask_ratio 0.15 \
    --positive_num 2 \
    --temperature 0.1 \
    --attn_head 8 \
    --dropout 0.3 \
    --num_workers 10 \
    --use_scheduler \
    --balanced \
    --root_path <root_path> \
    --log_dir <log_dir> \
    --pretrained_path <pretrained_path>