## Pre-training HPM

A typical command to pre-train ViT-B/16 with **single-node distributed training**, run the following on 1 node with 8 GPUs each:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 --node_rank 0 \
    main_pretrain.py \
    --batch_size 32 \
    --accum_iter 8 \
    --model mae_vit_base_patch16_dec512d8b \
    --input_size 224 \
    --token_size 14 \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /path/to/ImageNet/ \
    --output_dir  ./ourput_dir \
    --log_dir   ./log_dir \
    --experiment hpm_relative_in1k_ep200 \
    --learning_loss --relative
```

Please modify the ```/path/to/ImageNet/``` to your ```<data_path>```.
You can also move the txt files **IN1K/train.txt** and **IN1K/val.txt** to your imagenet root path.
Please find these files [here](https://github.com/implus/UM-MAE/tree/main/IN1K).

More scripts can be found in [scripts](https://github.com/Haochen-Wang409/HPM/tree/main/scripts).
