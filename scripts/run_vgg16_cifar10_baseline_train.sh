
train_baseline_vgg16()
{
    python main_vgg_cifar10.py --dataset cifar10 --depth 16 \
    --mode train \
    --save_path $1 \
    --total_epoches 160 \
    --schedule 40 80 120 \
    --gammas 0.2 0.2 0.2 \
    --lr 0.1 --decay 0.0005 --batch_size 256
}

# baseline training
train_baseline_vgg16 ./vgg16_cifar10_baseline/vgg16_round_1
train_baseline_vgg16 ./vgg16_cifar10_baseline/vgg16_round_2
train_baseline_vgg16 ./vgg16_cifar10_baseline/vgg16_round_3