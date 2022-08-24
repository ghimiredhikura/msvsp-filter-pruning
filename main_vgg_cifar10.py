
# Similarity and Magnitude based Variable Rate Filter Pruning for Efficient ConvNets

import sys, os, random, shutil, time, copy
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from models.vgg_cifar import vgg

#https://github.com/VainF/Torch-Pruning
import torch_pruning_tool.torch_pruning as tp

import torch
import torch.backends.cudnn as cudnn
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn as nn
import numpy as np
from utils.utils import AverageMeter, RecorderMeter, time_string
from utils.utils import convert_secs2time, get_ncc_sim_matrix, get_n_flops_

from scipy.spatial import distance

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--pretrain_path', type=str, default="./", help='..path of pre-trained model')
parser.add_argument('--pruned_path', type=str, default="./", help='..path of pruned model')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'prune'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epoches', type=int, default=160)
parser.add_argument('--method', type=str, choices=['norm', 'ncc', 'cos', 'eucl', 'mix'])
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--depth', default=16, type=int, help='depth of the neural network')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')

# compress rate
parser.add_argument('--rate_flop', type=float, default=0.342, help='This is flop reduction rate')

# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if args.cuda else 'cpu')
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main():

    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')

    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Pruned path: {}".format(args.pruned_path), log)
    print_log("Pruning Method: {}".format(args.method), log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    transform_train = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        dataset_train = CIFAR10(args.data_path, train=True, transform=transform_train, download=True)
        dataset_test = CIFAR10(args.data_path, train=False, transform=transform_test, download=True)
    elif args.dataset == 'cifar100':
        dataset_train = CIFAR100(args.data_path, train=True, transform=transform_train, download=True)
        dataset_test = CIFAR100(args.data_path, train=False, transform=transform_test, download=True)
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, 
                                              num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, 
                                              num_workers=args.workers)
    # subset of train dataset, 10000
    train_subset_index = random.sample(range(0, 49999), 10000)
    dataset_train_subset = torch.utils.data.Subset(dataset_train, train_subset_index)
    train_loader_subset = torch.utils.data.DataLoader(dataset_train_subset, batch_size=args.test_batch_size, shuffle=False, 
                                              num_workers=args.workers)

    model = vgg(dataset=args.dataset, depth=args.depth)
    print_log("=> Original network:\n {}".format(model), log, True)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cuda:
        criterion.cuda()

    if args.mode == 'prune':
        if os.path.isfile(args.pretrain_path):
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
        else:
            print("Check Pretrain file path: ", args.pretrain_path)
            return 

        if args.use_cuda:
            torch.cuda.empty_cache()
            model.cuda()

        pruned_model = prune(model, train_loader, train_loader_subset, test_loader, criterion, log)
        print_log("=> Network after pruning:\n {}".format(pruned_model), log, True)
        flops_pruned = get_n_flops_(model, img_size=(32, 32))
        print_log("Pruned Model Flops: %lf" % flops_pruned, log)

        train(pruned_model, train_loader, test_loader, criterion, log)

    if args.mode == 'train':
        if os.path.isfile(args.pretrain_path):
            print("Loading Model State Dict from: ", args.pretrain_path)
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
            args.start_epoch = pretrain['epoch']
        else:
            print("Training model from init.. ")

        if args.use_cuda:
            torch.cuda.empty_cache()
            model.cuda()

        train(model, train_loader, test_loader, criterion, log)

    elif args.mode == 'eval':       
        if os.path.isfile(args.pretrain_path):
            print_log("Load model from %s" % (args.pretrain_path), log)
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
            print('epoch:', pretrain['epoch'])
            print('depth:', pretrain['arch'])

            flops_original = get_n_flops_(model, img_size=(32, 32))
            print_log("Pretrain Model Flops: %lf" % flops_original, log)

            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()

            acc, loss = validate(test_loader, model, criterion)
            print_log("Pretrain Top@1: %0.4f, Loss: %0.4f" % (acc, loss), log)
        else:
            flops_original = get_n_flops_(model, img_size=(32, 32))
            print_log("Original Model Flops: %lf" % flops_original, log)

        if os.path.isfile(args.pruned_path):
            print_log("Load pruned model from %s" % (args.pruned_path), log)
            pruned = torch.load(args.pruned_path)
            print('epoch:', pruned['epoch'])
            print('depth:', pruned['arch'])
            model = pruned['state_dict']

            print_log("=> Pruned network :\n {}".format(model), log, True)

            flops_pruned = get_n_flops_(model, img_size=(32, 32))
            print_log("Pruned Model Flops: %lf" % flops_pruned, log)

            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()
            acc, loss = validate(test_loader, model, criterion)
            print_log("Pruned Top@1: %0.4f, Loss: %0.4f" % (acc, loss), log)
        
            flop_reduction_rate = (1.0 - flops_pruned / flops_original) * 100.0
            print_log("FLOPs Reduction Rate: %0.2lf%%" % flop_reduction_rate, log)

    log.close()

def prune(model, train_loader, train_loader_subset, test_loader, criterion, log):

    sub_inputs, sub_targets = get_train_subset_in_memory(train_loader_subset)

    with torch.no_grad():
        val_acc_1, val_loss = validate(test_loader, model, criterion)
        print_log("Before Prune - Val [Acc|Loss]: %.3f %% | %0.5f" % (val_acc_1, val_loss), log)

    flops_baseline = get_n_flops_(model, img_size=(32, 32))
    print_log("Baseline Model Flops: %lf" % flops_baseline, log)

    current_flop_reduction_rate = 0.0
    flop_reduction_rate_temp = 0.0
    step_prune_rate = 0.1
    filter_prune_limit_per_layer = 0.7
    flop_partition_rate = args.rate_flop/2.0

    model.cpu()
    layerwise_filter_count_org = get_conv_filter_count(model)
    DG = tp.DependencyGraph().build_dependency(model, torch.rand(1,3,32,32))

    method = args.method
    while current_flop_reduction_rate < args.rate_flop:
        small_loss = 100000000000.0
        small_loss_lindex = -1
        
        if args.method == 'mix':
            method = 'norm' if current_flop_reduction_rate < flop_partition_rate else 'eucl'

        for prune_conv_idx in layerwise_filter_count_org.keys():
            # model copy to prune 
            model_copy = copy.deepcopy(model)
            # prune 
            model_copy.cpu()
            DG_temp = tp.DependencyGraph().build_dependency(model_copy, torch.rand(1,3,32,32))

            success_flag = prune_ith_conv_layer(model_copy, DG_temp, prune_conv_idx, layerwise_filter_count_org[prune_conv_idx], 
                                    step_prune_rate, filter_prune_limit_per_layer, method)
            if not success_flag:
                continue

            # calc loss after prune 
            if args.use_cuda:
                torch.cuda.empty_cache()
                model_copy.cuda()
            with torch.no_grad():
                _, sample_loss = validate_fast(sub_inputs, sub_targets, model_copy, criterion)

            # store conv layer index with small loss 
            small_loss_lindex = prune_conv_idx if sample_loss < small_loss else small_loss_lindex
            small_loss = min(small_loss, sample_loss) 

        # prune selected layer with given prune rate
        prune_ith_conv_layer(model, DG, small_loss_lindex, layerwise_filter_count_org[small_loss_lindex],
                            step_prune_rate, filter_prune_limit_per_layer, method)

        flops_pruned = get_n_flops_(model, img_size=(32, 32))
        current_flop_reduction_rate = 1.0 - flops_pruned / flops_baseline
        print("[Pruning Method: %s] Flop Reduction Rate: %lf/%lf" % (method, current_flop_reduction_rate, args.rate_flop))
 
        if current_flop_reduction_rate - flop_reduction_rate_temp > 0.03:
            # train single epoch to recover recently pruned layer 
            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()

            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                        weight_decay=args.decay, nesterov=True)
            for epoch in range(0, 1):
                train_epoch(train_loader, model, criterion, optimizer)
            flop_reduction_rate_temp = current_flop_reduction_rate

    with torch.no_grad():
        val_acc_1, val_loss = validate(test_loader, model, criterion)
        print_log("After Prune, Before Finetune - Val [Acc|Loss]: %.3f %% | %0.5f" % (val_acc_1, val_loss), log)

    layerwise_filter_count_prune = {}
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            layerwise_filter_count_prune[idx] = m.weight.shape[0]

    print_log('Final Flop Reduction Rate: %.4lf' % current_flop_reduction_rate, log)
    print_log("Conv Filters Before Pruning: " + ''.join(str(layerwise_filter_count_org)), log)
    print_log("Conv Filters After Pruning: " + ''.join(str(layerwise_filter_count_prune)), log)    

    filter_prune_rate = {}
    for idx in layerwise_filter_count_org.keys():
        filter_prune_rate[idx] = 1.0 - float(layerwise_filter_count_prune[idx]/float(layerwise_filter_count_org[idx]))
    print_log("Layerwise Pruning Rate: " + ''.join(str(filter_prune_rate)), log)    

    return model

def get_conv_filter_count(model):
    conv_filter_count = {}
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight = m.weight.detach().numpy()
            conv_filter_count[idx] = int(weight.shape[0])
    return conv_filter_count

def prune_ith_conv_layer(model, DG, conv_idx, org_filter_count, step_prune_rate=0.1, 
                            max_pruning_rate=0.7, method='norm'):
    model.cpu()
    flag = False
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) and idx == conv_idx:
            nfilters = m.weight.detach().numpy().shape[0]
            last_prune_rate = 1 - float(nfilters) / float(org_filter_count)
            if last_prune_rate < max_pruning_rate:
                step_prune_count = int(nfilters * step_prune_rate + 0.5)
                prune_conv(DG, m, step_prune_count, method)
                flag = True
            break
    return flag

def get_train_subset_in_memory(train_loader_subset):
    sub_inputs = []
    sub_targets = []
    for _, (input, target) in enumerate(train_loader_subset):
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
        sub_targets.append(target)
        sub_inputs.append(input)
    return sub_inputs, sub_targets


def get_similar_matrix(weight_vec_after_norm, dist_type="eucl"):
    if dist_type == "eucl":
        similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
    elif dist_type == "cos":  # for cos similarity
        similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
        similar_matrix[np.isnan(similar_matrix)] = 1
        similar_matrix = 1 - similar_matrix
    elif dist_type == "ncc":
        similar_matrix = 1 - get_ncc_sim_matrix(weight_vec_after_norm)
        similar_matrix[np.isnan(similar_matrix)] = 1
        similar_matrix = 1 - similar_matrix
    
    return similar_matrix

def norm_prune(conv, amount=1):
    strategy = tp.strategy.L2Strategy()
    pruning_index = strategy(conv.weight, amount=amount)
    return pruning_index

def similarity_prune(conv, amount=0.2, method='eucl'):
    weight = conv.weight.detach().numpy()
    total_filters = weight.shape[0]
    weight = weight.reshape(total_filters, -1)
    num_prumed = int(total_filters * amount) if amount < 1.0 else amount
    similar_matrix = get_similar_matrix(weight, method)
    similar_sum = np.sum(similar_matrix, axis=1)
    pruning_index = np.argsort(similar_sum)[:num_prumed].tolist()

    return pruning_index

def prune_conv(DG, conv, amount, method):
    # get index of filters to be pruned        
    if method == "norm":
        pruning_index = norm_prune(conv, amount)
        # apply pruning
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    else:
        pruning_index = similarity_prune(conv, amount, method)
        # apply pruning
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()                    

def train(model, train_loader, test_loader, criterion, log):

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
            weight_decay=args.decay, nesterov=True)

    if args.use_cuda:
        torch.cuda.empty_cache()
        model.cuda()

    recorder = RecorderMeter(args.total_epoches)
    start_time = time.time()
    epoch_time = AverageMeter()
    best_accuracy = 0

    # Main loop
    for epoch in range(0, args.total_epoches):

        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        #current_learning_rate = adjust_learning_rate_vgg(optimizer, epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.total_epoches - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        # train for one epoch
        train_acc, train_los = train_epoch(train_loader, model, criterion, optimizer)

        # validate
        val_acc_2, val_los_2 = validate(test_loader, model, criterion)
        print("Epoch %d/%d [learning_rate=%lf] Val [Acc|Loss]: %.3f %% | %0.5f" % (epoch, args.total_epoches, current_learning_rate, val_acc_2, val_los_2))

        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)
        if recorder.max_accuracy(False) > best_accuracy:
            print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.total_epoches,
                                                                                    need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                100 - recorder.max_accuracy(False)), log)
            best_accuracy = recorder.max_accuracy(False)
            is_best = True

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.depth,
            'state_dict': model,
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

# train function (forward, backward, update)
def train_epoch(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        # Only need to do topk for highest k, reuse for the rest 
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        batch_size = target.size(0)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def validate_single_epoch(input, target, model, criterion, losses_m, top1_m, top5_m):
    if args.use_cuda:
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    losses_m.update(loss.item(), input.size(0))
    top1_m.update(prec1, input.size(0))
    top5_m.update(prec5, input.size(0))

    return losses_m, top1_m, top5_m    

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        losses, top1, top5 = validate_single_epoch(input, target, model, criterion, losses, top1, top5)
    return top1.avg, losses.avg

def validate_fast(inputs, targets, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for (input, target) in zip(inputs, targets):
        losses, top1, top5 = validate_single_epoch(input, target, model, criterion, losses, top1, top5)
    return top1.avg, losses.avg

def print_log(print_string, log, display=True):
    if display:
        print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_vgg(optimizer, epoch):
    if epoch < args.total_epoches * 0.20: 
        return args.lr
    
    if epoch in [args.total_epoches * 0.20, args.total_epoches * 0.40, args.total_epoches * 0.60, args.total_epoches * 0.80]:
        current_learning_rate = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            current_learning_rate = param_group['lr']
        return current_learning_rate

    if epoch >= args.total_epoches * 0.2:
        for param_group in optimizer.param_groups:
            current_learning_rate = param_group['lr']
        return current_learning_rate

if __name__=='__main__':
    main()