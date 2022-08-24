
IMAGENET_PATH="C:/ImageNet" 
BASELINE_SAVE_PATH="ResNet_ImageNet_baseline"

train_baseline_resnet_imagenet()
{
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode train \
    --arch $1 \
    --save_path $2 \
    --total_epoches 100 \
    --decay_epoch_step 30 \
    --workers 8 \
    --lr 0.1 --decay 0.0001 --batch_size 256
}

# resnet imagenet baseline training 
train_baseline_resnet_imagenet resnet18 $BASELINE_SAVE_PATH/resnet18
train_baseline_resnet_imagenet resnet34 $BASELINE_SAVE_PATH/resnet34
train_baseline_resnet_imagenet resnet50 $BASELINE_SAVE_PATH/resnet50