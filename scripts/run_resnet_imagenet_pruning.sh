
IMAGENET_PATH="C:/ImageNet"
BASELINE_MODEL_PATH="ResNet_ImageNet_baseline"
PRUNE_MODEL_DIR="ResNet_ImageNet_pruned"

prune_baseline_resnet_imagenet()
{
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode prune \
    --arch $1 \
    --pretrain_path $2 \
    --save_path $3 \
    --method mix \
    --rate_flop $4 \
    --total_epoches 100 \
    --start_epoch 20 \
    --recover_epoch 2 \
    --retrain_step_flop 0.03 \
    --decay_epoch_step 30 \
    --workers 8 \
    --lr 0.01 --decay 0.0001 --batch_size 256
}

# pruning, resnet18
prune_baseline_resnet_imagenet resnet18 $BASELINE_MODEL_PATH $PRUNE_MODEL_DIR/resnet18 0.418

# pruning, resnet34
prune_baseline_resnet_imagenet resnet34 $BASELINE_MODEL_PATH $PRUNE_MODEL_DIR/resnet34 0.411

# pruning, resnet50
prune_baseline_resnet_imagenet resnet50 $BASELINE_MODEL_PATH $PRUNE_MODEL_DIR/resnet50 0.422