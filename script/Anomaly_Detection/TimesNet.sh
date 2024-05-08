torchrun \
    --standalone \
    --nproc_per_node=1 \
    run.py \
    --task_name anomaly_detection \
    --model TimesNet \
    --learning_rate 1e-3 \
    --patience 0 \
    --d_hidden 8 \
    --num_kernels 6 \
    --d_model 16 \
    --e_layers 3 \
    --dropout 0.4 \
    --top_k 3 \
    --num_epochs 60 \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --use_scheduler \
    --num_workers 10 \
    --dataset TUAB \
    --input_len 12 \
    --root_path <root_path> \
    --log_dir <log_dir> \
    --normalize