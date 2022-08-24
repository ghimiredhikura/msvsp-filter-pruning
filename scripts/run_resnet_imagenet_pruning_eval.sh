

IMAGENET_PATH="C:/ImageNet"
PRUNED_MODEL_DIR="pruned_models/ImageNet-RESNET-PRUNED"
BASELINE_MODEL_DIR="baseline_models/ImageNet-RESNET-BASELINE"
RESULT_SAVE_PATH="./eval_pruning_imagenet_resnet"

eval_prune_resnet_imagenet()
{
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode eval \
    --arch $1 \
    --pretrain_path $2 \
    --pruned_path $3 \
    --save_path $4 \
    --workers 8 \
    --batch_size 256
}

# eval resnet18, flop 0.41
BASELINE_MODEL=$BASELINE_MODEL_DIR/resnet18/resnet18.model_best.pth.tar
PRUNED_MODEL=$PRUNED_MODEL_DIR/resnet18_flop[0.41]/resnet18.model_best.pth.tar
eval_prune_resnet_imagenet resnet18 $BASELINE_MODEL $PRUNED_MODEL $RESULT_SAVE_PATH/resnet18

# eval resnet18, flop 0.45
BASELINE_MODEL=$BASELINE_MODEL_DIR/resnet18/resnet18.model_best.pth.tar
PRUNED_MODEL=$PRUNED_MODEL_DIR/resnet18_flop[0.45]/resnet18.model_best.pth.tar
eval_prune_resnet_imagenet resnet18 $BASELINE_MODEL $PRUNED_MODEL $RESULT_SAVE_PATH/resnet18

# eval resnet34, flop 0.41
BASELINE_MODEL=$BASELINE_MODEL_DIR/resnet34/resnet34.model_best.pth.tar
PRUNED_MODEL=$PRUNED_MODEL_DIR/resnet34_flop[0.41]/resnet34.model_best.pth.tar
eval_prune_resnet_imagenet resnet34 $BASELINE_MODEL $PRUNED_MODEL $RESULT_SAVE_PATH/resnet34

# eval resnet34, flop 0.45
BASELINE_MODEL=$BASELINE_MODEL_DIR/resnet34/resnet34.model_best.pth.tar
PRUNED_MODEL=$PRUNED_MODEL_DIR/resnet34_flop[0.45]/resnet34.model_best.pth.tar
eval_prune_resnet_imagenet resnet34 $BASELINE_MODEL $PRUNED_MODEL $RESULT_SAVE_PATH/resnet34

# eval resnet50, flop 0.42
BASELINE_MODEL=$BASELINE_MODEL_DIR/resnet50/resnet50.model_best.pth.tar
PRUNED_MODEL=$PRUNED_MODEL_DIR/resnet50_flop[0.42]/resnet50.model_best.pth.tar
eval_prune_resnet_imagenet resnet50 $BASELINE_MODEL $PRUNED_MODEL $RESULT_SAVE_PATH/resnet50

# eval resnet50, flop 0.53
BASELINE_MODEL=$BASELINE_MODEL_DIR/resnet50/resnet50.model_best.pth.tar
PRUNED_MODEL=$PRUNED_MODEL_DIR/resnet50_flop[0.53]/resnet50.model_best.pth.tar
eval_prune_resnet_imagenet resnet50 $BASELINE_MODEL $PRUNED_MODEL $RESULT_SAVE_PATH/resnet50