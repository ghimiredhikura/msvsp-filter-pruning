
prune_baseline_vgg16()
{
    python main_vgg_cifar10.py --dataset cifar10 --depth 16 \
    --mode prune \
    --pretrain_path $1 \
    --save_path $2 \
    --method $3 \
    --rate_flop $4 \
    --total_epoches 160 \
    --schedule 40 80 120 \
    --gammas 0.2 0.2 0.2 \
    --lr 0.01 --decay 0.0005 --batch_size 256
}

BASELINE_PATH="./baseline_models/CIFAR10-VGG16_BASELINE"
PRUNING_PATH="./vgg16_pruned"
FLOP_REDUCTION_RATE="0.342"

prune_baseline_vgg16 $BASELINE_PATH/vgg16_round_1/model_best.pth.tar $PRUNING_PATH/vgg16_round_1 mix $FLOP_REDUCTION_RATE
prune_baseline_vgg16 $BASELINE_PATH/vgg16_round_2/model_best.pth.tar $PRUNING_PATH/vgg16_round_2 mix $FLOP_REDUCTION_RATE
prune_baseline_vgg16 $BASELINE_PATH/vgg16_round_3/model_best.pth.tar $PRUNING_PATH/vgg16_round_3 mix $FLOP_REDUCTION_RATE