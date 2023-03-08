## Fine-tuning HPM

A typical command to fine-tune ViT-B/16 with **single-node distributed training**, run the following on 1 node with 8 GPUs each:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 --node_rank 0 \
    main_finetune.py \
    --batch_size 48 \
    --accum_iter 1 \
    --model vit_base_patch16 \
    --finetune /path/to/checkpoint/ \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 5e-4 --layer_decay 0.8 --weight_decay 0.05 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path /path/to/ImageNet/ \
    --dataloader_type nori --nb_classes 1000 \
    --output_dir  ./output_dir/finetune \
    --log_dir  ./log_dir/finetune \
    --experiment hpm_in1k_ep100
```
Please modify ```/path/to/ImageNet/``` to your ```<data_path>````.
You can also move the txt files **IN1K/train.txt** and **IN1K/val.txt** to your imagenet root path.
Please find these files [here](https://github.com/implus/UM-MAE/tree/main/IN1K).

More scripts can be found in [scripts](https://github.com/Haochen-Wang409/HPM/tree/main/scripts).
