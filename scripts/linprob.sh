python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes 1 --node_rank 0 \
    main_linprob.py \
    --batch_size 512 \
    --model vit_base_patch16 \
    --finetune /path/to/checkpoint/ \
    --epochs 100 \
    --blr 1e-3 --weight_decay 0.0 \
    --dist_eval \
    --data_path /path/to/ImageNet/ \
    --nb_classes 1000 \
    --log_dir   ./log_dir/linprob