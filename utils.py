
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import random
from torch.backends import cudnn
import math
from pyhessian import hessian
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn
import copy
from opacus.accountants.utils import get_noise_multiplier
##############################################################################
# Tools
##############################################################################


def init_model(num_id, model_type, args):
    if args.dataset == 'OCT':
        from classnames import OCT_Concepts
        classnames = OCT_Concepts
    elif args.dataset == 'Kvasir':
        from classnames import Kvasir_Concepts
        classnames = Kvasir_Concepts
    else:
        assert False
    num_classes = args.num_classes
    if model_type == 'CNN':
        if args.dataset == 'cifar10':
            model = cnn.CNNCifar10()
        else:
            model = cnn.CNNCifar100()
    # resnet18 for imagenet
    elif model_type == 'ResNet18':
        model = resnet.ResNet18(num_classes)
    elif model_type == 'ResNet20':
        model = resnet.ResNet20(num_classes)
    elif model_type == 'ResNet56':
        model = resnet.ResNet56(num_classes)
    elif model_type == 'ResNet110':
        model = resnet.ResNet110(num_classes)
    elif model_type == 'WRN56_2':
        model = resnet.WRN56_2(num_classes)
    elif model_type == 'WRN56_4':
        model = resnet.WRN56_4(num_classes)
    elif model_type == 'WRN56_8':
        model = resnet.WRN56_8(num_classes)
    elif model_type == 'DenseNet121':
        model = densenet.DenseNet121(num_classes)
    elif model_type == 'DenseNet169':
        model = densenet.DenseNet169(num_classes)
    elif model_type == 'DenseNet201':
        model = densenet.DenseNet201(num_classes)
    elif model_type == 'MLP':
        model = cnn.MLP()
    elif model_type == 'LeNet5':
        model = cnn.LeNet5() 
    elif model_type == 'CLIP':
        if args.method == 'DP-DyLoRA':
            from models_dict.CLIP_DyLoRA import local_clip_model
        else:
            from models_dict.CLIP import local_clip_model
        model, text_features = local_clip_model(args, num_id, args.lora_r, classnames)
        return model, text_features

    return model

def init_optimizer(num_id, model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.local_wd_rate)

    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training function
##############################################################################

def generate_matchlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))
    

##############################################################################
# Validation function
##############################################################################
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def compute_metrics(pre, gt): #D, H, W
    #在测试集每类数量相同的情况下，多分类度量指标Accuracy=macro-Recall=micro-F1；
    #在测试集每类数量不相同的情况下，多分类度量指标Accuracy= micro-F1。
    pred = pre.cpu().numpy()
    gt = gt.cpu().numpy()
    
    acc=accuracy_score(gt, pred)
    recall=recall_score(gt, pred, average='macro')
    prec = precision_score(gt, pred, average='macro')
    f1 = f1_score(gt, pred, average='macro')

    return acc, recall, prec, f1

def compute_auc(pre_scores, gt, num_classes = 8):

    pre_scores = pre_scores.cpu().numpy()
    gt = gt.cpu().numpy()
    
    gt_one_hot = np.eye(num_classes)[gt]
    auc_score = roc_auc_score(gt_one_hot, pre_scores)
    
    return auc_score
    

def validate(args, node, which_dataset = 'validate'):
    '''
    Generally, 'validate' refers to the local datasets of clients and 'local' refers to the server's testset.
    '''
    node.model.cuda().eval() 
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    elif which_dataset == 'test':
        test_loader = node.test_set
    else:
        raise ValueError('Undefined...')

    best_list = []
    if args.method == 'DP-DyLoRA':
        best_performce = -1
        if args.lora_r == 2: new_ranks = [1, 2]
        if args.lora_r == 4: new_ranks = [1, 2, 4]
        if args.lora_r == 8: new_ranks = [1, 2, 4, 8]
        if args.lora_r == 16: new_ranks = [1, 2, 4, 8, 16]
        
        for new_rank in new_ranks:
            node.model.set_rank(new_rank, frozen=False)
            with torch.no_grad():
                preds = []
                targets = []
                pred_scores = []
                for idx, (data, target) in enumerate(test_loader):
                    data, target = data.cuda(), target.cuda()
                    image_features = node.model(data)
                    text_features = node.text_features
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    output = image_features @ text_features.T # B 512   C 512
        
                    pred = output.argmax(dim=1)
                    pred_scores.append(output.softmax(dim=1)) # B, C
                    preds.append(pred)
                    targets.append(target.view_as(pred))
        
                pred_scores = torch.cat(pred_scores)
                preds = torch.cat(preds)
                targets = torch.cat(targets)
                acc, recall, prec, f1 = compute_metrics(preds, targets)
                
                auc = compute_auc(pred_scores, targets, num_classes=args.num_classes)
                if best_performce < acc + recall + prec + f1 + auc:
                    best_performce = acc + recall + prec + f1 + auc
                    best_list = [acc, recall, prec, f1, auc]
    else:
        with torch.no_grad():
            preds = []
            targets = []
            pred_scores = []
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                image_features = node.model(data)
                text_features = node.text_features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                output = image_features @ text_features.T # B 512   C 512
    
                pred = output.argmax(dim=1)
                pred_scores.append(output.softmax(dim=1)) # B, C
                preds.append(pred)
                targets.append(target.view_as(pred))
    
            pred_scores = torch.cat(pred_scores)
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            acc, recall, prec, f1 = compute_metrics(preds, targets)
            
            auc = compute_auc(pred_scores, targets, num_classes=args.num_classes)
            best_list = [acc, recall, prec, f1, auc]

    return best_list[0]*100, best_list[1]*100, best_list[2]*100, best_list[3]*100, best_list[4]*100


def compute_noise_multiplier(args, target_epsilon, target_delta, global_epoch, local_epoch, batch_size, client_data_sizes):
    total_dataset_size = sum(client_data_sizes)
    sample_rate = batch_size / total_dataset_size
    total_steps = sum([global_epoch * local_epoch * (client_data_size / batch_size) for client_data_size in client_data_sizes])

    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps, 
        accountant="rdp"
    )

    return noise_multiplier