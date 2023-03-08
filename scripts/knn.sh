python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes 1 --node_rank 0 \
    main_knn.py \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune /path/to/checkpoint/ \
    --data_path /path/to/ImageNet/ \
    --nb_classes 1000