
# Similarity and Magnitude based Variable Rate Filter Pruning for Efficient ConvNets

import sys, os, random, shutil, time, copy
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# https://github.com/VainF/Torch-Pruning
import torch_pruning_tool.torch_pruning as tp

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets 
from torchvision import transforms
import numpy as np
from utils.utils import AverageMeter, RecorderMeter, time_string
from utils.utils import convert_secs2time, get_ncc_sim_matrix, get_n_flops_

import models
import models.imagenet_resnet as resnet
from scipy.spatial import distance

parser = argparse.ArgumentParser()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser.add_argument("--data_path", type=str, help='imagenet dataset path')
parser.add_argument('--pretrain_path', default='./', type=str, help='..path of pre-trained model')
parser.add_argument('--pruned_path', default='./', type=str, help='..path of pruned model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--mode', type=str, default="eval", choices=['train', 'eval', 'prune'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epoches', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--recover_epoch', type=int, default=1)
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--method', type=str, choices=['norm', 'ncc', 'cos', 'eucl', 'mix'])
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--decay_epoch_step', default=30, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# compress rate
parser.add_argument('--rate_flop', type=float, default=0.342, help='This is flop reduction rate')

# retrain after step flop rate
parser.add_argument('--retrain_step_flop', type=float, default=0.03, help='retrain every 3% flop reduction')

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

def print_log(print_string, log, display=True):
    if display:
        print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def main():

    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')

    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Pruned path: {}".format(args.pruned_path), log)
    print_log("Pruning Method: {}".format(args.method), log)

    # Data loading code
    if args.data_path is not None:
        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, sampler=None)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

        # subset of train dataset, 5000
        subset_index = random.sample(range(0, 149999), 10000)
        dataset_subset = torch.utils.data.Subset(train_dataset, subset_index)
        loader_subset = torch.utils.data.DataLoader(dataset_subset, batch_size=args.batch_size, shuffle=False, 
                                                num_workers=args.workers)

    if args.data_path is None:
        if args.mode == 'train' or args.mode == 'prune':
            print("ImageNet dataset path is not provided!")
            return

    # create model
    print_log("=> Creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=False)
    print_log("=> Original Model : {}".format(model), log)
    print_log("=> Parameter : {}".format(args), log)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        criterion.cuda()

    if args.mode == 'prune':
        if os.path.isfile(args.pretrain_path):
            print("Loading Model State Dict from: ", args.pretrain_path)
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
        else:
            print("Check Pretrain file path: ", args.pretrain_path)
            return

        if args.use_cuda:
            torch.cuda.empty_cache()
            model.cuda()

        pruned_model = prune(model, train_loader, loader_subset, test_loader, criterion, log)
        print_log("=> Model [After Pruning]:\n {}".format(pruned_model), log)
        flops_pruned = get_n_flops_(model, img_size=(224, 224))
        print_log("Pruned Model Flops: %lf" % flops_pruned, log)

        train(pruned_model, train_loader, test_loader, criterion, args.start_epoch, args.total_epoches, log)

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

        train(model, train_loader, test_loader, criterion, args.start_epoch, args.total_epoches, log)

    elif args.mode == 'eval':       
        if os.path.isfile(args.pretrain_path):
            print("Loading Pretrain Model from: ", args.pretrain_path)
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
            print('epoch:', pretrain['epoch'])
            print('depth:', pretrain['arch'])

            flops_original = get_n_flops_(model, img_size=(224, 224))
            print_log("Pretrain Model Flops: %lf" % flops_original, log)

            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()

            if args.data_path is not None:
                val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model, criterion)
                print_log("Pretrain Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log)
        else:
            flops_original = get_n_flops_(model, img_size=(224, 224))
            print_log("Original Model Flops: %lf" % flops_original, log)

        if os.path.isfile(args.pruned_path):
            print("Load Pruned Model from %s" % (args.pruned_path))
            pruned = torch.load(args.pruned_path)
            model_pruned = pruned['state_dict']
            print_log("=> Pruned network :\n {}".format(model_pruned), log, True)

            if args.use_cuda:
                torch.cuda.empty_cache()
                model_pruned.cuda()

            flops_pruned = get_n_flops_(model_pruned, img_size=(224, 224))
            print_log("Pruned Model Flops: %lf" % flops_pruned, log)
            
            if args.data_path is not None:
                val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model_pruned, criterion)
                print_log("Pruned Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log)

            flop_reduction_rate = (1.0 - flops_pruned / flops_original) * 100.0
            print_log("Flop Reduction Rate: %0.2lf%%" % (flop_reduction_rate), log)

    log.close()

def prune(model, train_loader, loader_subset, test_loader, criterion, log):

    sub_inputs, sub_targets = get_train_subset_in_memory(loader_subset)

    with torch.no_grad():
        val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model, criterion)
        print_log("Before Prune, Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log) 

    flops_baseline = get_n_flops_(model, img_size=(224, 224))

    current_flop_reduction_rate = 0.0
    flop_reduction_rate_temp = 0.0
    step_prune_rate = 0.1
    filter_prune_limit_per_layer = 0.7
    flop_partition_rate = 1*args.rate_flop/2.0

    model.eval()
    layerwise_filter_count_org = get_conv_filter_count(model)
    print(layerwise_filter_count_org)
    
    print("Start Pruning ...")    
    method = args.method

    while current_flop_reduction_rate < args.rate_flop:
        small_loss = 100000000000.0
        small_loss_lindex = ''
        
        if args.method == 'mix':
            method = 'norm' if current_flop_reduction_rate < flop_partition_rate else 'eucl'

        for prune_conv_idx in layerwise_filter_count_org.keys():
            # model copy to prune 
            model_copy = copy.deepcopy(model)
            # prune 
            model_copy.eval()
            DG_temp = tp.DependencyGraph().build_dependency(model_copy, torch.randn(1,3,224,224))

            step_prune_count = prune_single_conv_layer(model_copy, DG_temp, prune_conv_idx, layerwise_filter_count_org[prune_conv_idx], 
                                    step_prune_rate, filter_prune_limit_per_layer, method)
            if step_prune_count == 0:
                continue

            # calc loss after prune 
            if args.use_cuda:
                torch.cuda.empty_cache()
                model_copy.cuda()
            with torch.no_grad():
                acc, sample_loss = validate_fast(sub_inputs, sub_targets, model_copy, criterion)

            # store conv layer index with small loss 
            if sample_loss < small_loss:
                small_loss_lindex = prune_conv_idx
                small_loss = sample_loss

        # prune selected layer with given prune rate
        DG = tp.DependencyGraph().build_dependency(model, torch.randn(1,3,224,224))
        step_prune_count = prune_single_conv_layer(model, DG, small_loss_lindex, layerwise_filter_count_org[small_loss_lindex],
                            step_prune_rate, filter_prune_limit_per_layer, method)

        flops_pruned = get_n_flops_(model, img_size=(224, 224))
        current_flop_reduction_rate = 1.0 - flops_pruned / flops_baseline
        print_log("[Pruning Method: %s] Flop Reduction Rate: %lf/%lf [Pruned %d filters from %s]" % (method, 
                    current_flop_reduction_rate, args.rate_flop, step_prune_count, small_loss_lindex), log)

        if current_flop_reduction_rate - flop_reduction_rate_temp > args.retrain_step_flop:
            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()

            # train 
            train(model, train_loader, test_loader, criterion, 0, args.recover_epoch, log)

            best_path = os.path.join(args.save_path, "resnet18.model_best.pth.tar")
            print("Loading Best Model State Dict from: ", best_path)
            pretrain = torch.load(best_path)
            model = pretrain['state_dict']

            with torch.no_grad():
                val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model, criterion)
                print_log("Best Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log) 

            flop_reduction_rate_temp = current_flop_reduction_rate

    layerwise_filter_count_prune = get_conv_filter_count(model)
    print_log('Final Flop Reduction Rate: %.4lf' % current_flop_reduction_rate, log)
    print_log("Conv Filters Before Pruning: " + ''.join(str(layerwise_filter_count_org)), log)
    print_log("Conv Filters After Pruning: " + ''.join(str(layerwise_filter_count_prune)), log)    

    filter_prune_rate = {}
    for idx in layerwise_filter_count_org.keys():
        filter_prune_rate[idx] = 1.0 - float(layerwise_filter_count_prune[idx]/float(layerwise_filter_count_org[idx]))
    print_log("Layerwise Pruning Rate: " + ''.join(str(filter_prune_rate)), log)    

    # save pruned model before finetuning
    save_checkpoint({
        'epoch': 0,
        'arch': args.arch,
        'state_dict': model,
    }, False, args.save_path, '%s.init.prune.pth.tar' % (args.arch) )

    return model

def get_conv_filter_count(model):
    conv_filter_count = {}
    if args.arch == 'resnet18' or args.arch == 'resnet34':
        for idx, m in enumerate(model.modules()):
            if isinstance(m, resnet.BasicBlock):
                conv_filters = m.conv1.weight.shape[0]
                key = 'conv1_%d' % (idx)
                conv_filter_count[key] = int(conv_filters)
                conv_filters = m.conv2.weight.shape[0]
                key = 'conv2_%d' % (idx)
                conv_filter_count[key] = int(conv_filters)
    elif args.arch == 'resnet50':
        for idx, m in enumerate(model.modules()):
            if isinstance(m, resnet.Bottleneck):
                conv_filters = m.conv1.weight.shape[0]
                key = 'conv1_%d' % (idx)
                conv_filter_count[key] = int(conv_filters)
                conv_filters = m.conv2.weight.shape[0]
                key = 'conv2_%d' % (idx)
                conv_filter_count[key] = int(conv_filters)
                conv_filters = m.conv3.weight.shape[0]
                key = 'conv3_%d' % (idx)
                conv_filter_count[key] = int(conv_filters)
    return conv_filter_count

def prune_single_conv_layer(model, DG, conv_key, org_filter_count, step_prune_rate=0.1, 
                            max_pruning_rate=0.7, method='norm'):
    #model.cpu()
    step_prune_count = 0
    key_name = conv_key.split('_')[0]
    key_id = int(conv_key.split('_')[1])

    if args.arch == 'resnet18' or args.arch == 'resnet34': 
        for idx, m in enumerate(model.modules()):
            if isinstance(m, resnet.BasicBlock) and idx == key_id:
                if key_name == 'conv1': conv_module = m.conv1
                elif key_name == 'conv2': conv_module = m.conv2
                else: print("Warning: %s do not exist in network", key_name)
                #nfilters = conv_module.weight.detach().numpy().shape[0]
                nfilters = conv_module.weight.shape[0]
                last_prune_rate = 1 - float(nfilters) / float(org_filter_count)
                if last_prune_rate < max_pruning_rate:
                    step_prune_count = int(nfilters * step_prune_rate + 0.5)
                    prune_conv(DG, conv_module, step_prune_count, method)
                break
    elif args.arch == 'resnet50':
        for idx, m in enumerate(model.modules()):
            if isinstance(m, resnet.Bottleneck) and idx == key_id:
                if key_name == 'conv1': conv_module = m.conv1
                elif key_name == 'conv2': conv_module = m.conv2
                elif key_name == 'conv3': conv_module = m.conv3
                else: print("Warning: %s do not exist in network", key_name)
                #nfilters = conv_module.weight.detach().numpy().shape[0]
                nfilters = conv_module.weight.shape[0]
                last_prune_rate = 1 - float(nfilters) / float(org_filter_count)
                if last_prune_rate < max_pruning_rate:
                    step_prune_count = int(nfilters * step_prune_rate + 0.5)
                    prune_conv(DG, conv_module, step_prune_count, method)
                break
    return step_prune_count

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
    weight = conv.weight.detach().cpu().numpy()
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

def train(model, train_loader, test_loader, criterion, start_epoch, total_epoches, log):

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
    for epoch in range(start_epoch, total_epoches):

        current_learning_rate = adjust_learning_rate(optimizer, epoch)
        #current_learning_rate = adjust_learning_rate_vgg(optimizer, epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.total_epoches - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        # train for one epoch
        train_acc, train_los = train_epoch(train_loader, model, criterion, optimizer, epoch, log)

        # validate
        val_acc_top1, val_acc_top5, val_los = validate(test_loader, model, criterion)
        print_log("Epoch %d/%d [learning_rate=%lf] Val [Acc@1=%0.3f, Acc@5=%0.3f | Loss= %0.5f" % 
                (epoch, args.total_epoches, current_learning_rate, val_acc_top1, val_acc_top5, val_los), log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc_top1)
        if recorder.max_accuracy(False) > best_accuracy:
            print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.total_epoches,
                                                                                    need_time, current_learning_rate) \
                + ' [Best : Acc@1={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                100 - recorder.max_accuracy(False)), log)
            best_accuracy = recorder.max_accuracy(False)
            is_best = True

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model,
        }, is_best, args.save_path, '%s.checkpoint.pth.tar' % (args.arch))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

# train function (forward, backward, update)
def train_epoch(train_loader, model, criterion, optimizer, epoch, log):
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

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5), log)

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
    return top1.avg, top5.avg, losses.avg

def validate_fast(inputs, targets, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for (input, target) in zip(inputs, targets):
        losses, top1, top5 = validate_single_epoch(input, target, model, criterion, losses, top1, top5)
    return top1.avg, losses.avg

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, '%s.model_best.pth.tar' % (args.arch))
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every args.decay_epoch_step epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__=='__main__':
    main()