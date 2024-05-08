torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name classification \
    --model DCRNN \
    --graph_type distance \
    --normalize \
    --use_fft \
    --learning_rate 1e-3 \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --patience 0 \
    --pretrained_path <pretrained_path>\
    --dropout 0.0 \
    --use_scheduler \
    --weight_decay 1e-4 \
    --num_workers 10 \
    --num_epochs 60 \
    --input_len 12