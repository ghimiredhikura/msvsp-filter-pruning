
eval_pruning_vgg16()
{
    python main_vgg_cifar10.py --dataset cifar10 --depth 16 \
    --mode eval \
    --depth 16 \
    --pretrain_path $1 \
    --pruned_path $2 \
    --save_path $3 
}

BASELINE_PATH="./baseline_models/CIFAR10-VGG16_BASELINE"
PRUNED_PATH="./pruned_models/CIFAR10-VGG16-PRUNED"
RESULT_SAVE_PATH="./eval_pruning_cifar10_vgg16"

eval_pruning_vgg16 $BASELINE_PATH/vgg16_round_1/model_best.pth.tar $PRUNED_PATH/vgg16_round_1_flop_0.342/model_best.pth.tar $RESULT_SAVE_PATH/vgg16_round_1
eval_pruning_vgg16 $BASELINE_PATH/vgg16_round_2/model_best.pth.tar $PRUNED_PATH/vgg16_round_2_flop_0.342/model_best.pth.tar $RESULT_SAVE_PATH/vgg16_round_2
eval_pruning_vgg16 $BASELINE_PATH/vgg16_round_3/model_best.pth.tar $PRUNED_PATH/vgg16_round_3_flop_0.342/model_best.pth.tar $RESULT_SAVE_PATH/vgg16_round_3