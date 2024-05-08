torchrun \
    --standalone \
    --nproc_per_node=4 \
    run.py \
    --task_name ssl \
    --model DCRNN \
    --use_curriculum_learning \
    --graph_type distance \
    --learning_rate 5e-4 \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --normalize \
    --use_fft \
    --data_augment \
    --num_epochs 100 \
    --dropout 0.2 \
    --patience 0 \
    --use_scheduler \
    --weight_decay 1e-4 \
    --num_workers 10 \
    --loss_fn mae \
    --dataset TUAB \
    --root_path <root_path> \
    --log_dir <log_dir> 

