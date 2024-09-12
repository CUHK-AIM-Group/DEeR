
import time
import torch
from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
import numpy as np
import os
import torch.nn as nn
import copy
import torch.optim as optim
import torch.nn.functional as F
import math
from pyhessian import hessian
from server_funct import *
#import wandb
from client_funct import *
import pprint
import argparse
import warnings
from utils import compute_noise_multiplier

warnings.filterwarnings('ignore')
np.set_printoptions(precision=7, suppress=True)
'''
ViT-L/14 768
ViT-B/16 512
ViT-B/32 512
'''
def generate_matchlist(node_num, ratio = 0.5):
    candidate_list = [i for i in range(node_num)]
    select_num = int(ratio * node_num)
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

if __name__ == '__main__':
    
    import gc 
    gc.collect()  # 清理内存

    #args = args_parser()
    ##### Exp settings #####
    ##### change it for different exps #####
    #args.client_method = 'fedetf'
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/Retinal_OCT-C8/',
                        help='./data/Retinal_OCT-C8/, ./data/kvasir-dataset-v2-processed-224')
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for iid, and 0 for noniid (dir. sampling)')
    parser.add_argument('--batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, 
                    help="dirichlet_alpha")
    parser.add_argument('--num_classes', type=int, default=8, 
                        help="num_classes")
    
    # System
    parser.add_argument('--device', type=str, default='0',
                        help="cuda device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=20, 
                        help="Number of nodes") 
    parser.add_argument('--T', type=int, default=200, 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=3, 
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='cifar10', #Kvasir
                        help="Type of algorithms:{mnist, cifar10,cifar100, fmnist, tinyimagenet}") 
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet20, ResNet18}')
    parser.add_argument('--random_seed', type=list, default=[10, 100, 1000], #
                        help="random seed for the whole experiment")
    parser.add_argument('--exp_name', type=str, default='FirstTable',
                        help="experiment name")
                        
    # Client function
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--lr', type=float, default=0.04,  
                        help='learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--mu', type=float, default=0.001,
                        help="proximal term mu")

    #add meilu
    parser.add_argument('--method', type=str, default='LORA',
                        help="method") # LORA,  FFA-LoRA, DEeR, DP-DyLoRA
    parser.add_argument('--lora_r', type=int, default=1,
                        help="lora_r") #CLIP, BiomedCLIP
    # DP noise
    parser.add_argument('--is_DP', type=int, default=0,
                        help="is_DP") # whether use DP
    parser.add_argument('--C', type=float, default=2, # 2 5 10
                        help='the threshold of clipping in DP')
    parser.add_argument('--epsilon', type=float, default=1.,
                        help='the standard deviation of client-level DP noise')
    
    parser.add_argument('--module1', type=int, default=1,
                        help="module1")
    parser.add_argument('--module2', type=int, default=1,
                        help="module2")
    
    args = parser.parse_args()
    
    all_acc, all_recall, all_prec, all_f1, all_auc = [],[],[],[],[]
    
    if args.dataset == 'Kvasir':
        batchsize = 128
    elif args.dataset == 'OCT':
        batchsize = 512
    args.batchsize = batchsize//args.node_num
    
    random_seeds = args.random_seed
    lr = args.lr
    for random_seed in random_seeds:
        args.random_seed = random_seed
        args.lr = lr
        print('starting run seed', args.random_seed)
        
        setup_seed(random_seed)
    
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The starting time ：{}'.format(now), flush=True)
        
        pprint(vars(args))
    
        select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]

        setting_name =  args.exp_name + '_' + args.method + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) + '_dir' + str(args.dirichlet_alpha) +'_E'+ str(args.E) \
        + '_' + args.server_method + '_' + args.client_method + '_seed' + str(args.random_seed)
    
        root_path = './'
        output_path = 'results/'
        if not os.path.exists(os.path.join(root_path, output_path)):
            os.makedirs(os.path.join(root_path, output_path))

        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
        data = Data(args)

        size_weights = [1./args.node_num] * args.node_num
        print('size-based weights',size_weights, flush=True)
        central_node = Node(args, -1, train_loader = None, val_loader=data.val_loader, test_loader=data.test_loader)
        # initialize the client nodes
        client_nodes = {}
        for i in range(args.node_num): 
            client_nodes[i] = Node(args, i, train_loader=data.train_loaders[i], val_loader=None, test_loader=None) 
            client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
            client_nodes[i].text_features = copy.deepcopy(central_node.text_features.data)

        test_acc_recorder = []
        best_val_acc = 0
        best_test_acc = 0
        best_test_recall=0
        best_test_prec=0
        best_test_f1=0
        best_test_auc=0
        print(setting_name, flush=True)
        ##################################
        noise_multiplier = None
        if args.is_DP:
            target_epsilon = args.epsilon
            target_delta = 1./args.node_num
            global_epoch = args.T
            local_epoch = args.E
            batch_size = args.batchsize
            sample_sizes = []
            for i in range(args.node_num): 
                sample_sizes.append(len(data.train_loader[i]))
            client_data_sizes = sample_sizes
            noise_multiplier = compute_noise_multiplier(args, target_epsilon, target_delta, global_epoch, local_epoch, batch_size, client_data_sizes)
            print(f'noise_multiplier : {noise_multiplier}')
        #################################
        for rounds in range(0, args.T):
            print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1), flush=True)
            #lr_scheduler(rounds, client_nodes, args)
            # Client selection
            select_list = select_list_recorder[rounds]
    
            if args.method == 'DEeR':
                if args.module1 == 1:
                    for i in range(len(client_nodes)):
                        for name, param in client_nodes[i].model.named_parameters():
                            if 'linear_a' in name:
                                param.requires_grad = False
                            if 'linear_b' in name:
                                param.requires_grad = True
                        client_nodes[i].optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, client_nodes[i].model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
                    client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
                    print('Train loss is {:.5f}'.format(train_loss), flush=True)
                    central_node = Server_update(args, central_node, client_nodes, select_list, size_weights,noise_multiplier, open_ab = 'b')
                    
                    for i in range(len(client_nodes)):
                        for name, param in client_nodes[i].model.named_parameters():
                            if 'linear_a' in name:
                                param.requires_grad = True
                            if 'linear_b' in name:
                                param.requires_grad = False
                        client_nodes[i].optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, client_nodes[i].model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
                    client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
                    print('Train loss is {:.5f}'.format(train_loss), flush=True)
                    central_node = Server_update(args, central_node, client_nodes, select_list, size_weights,noise_multiplier, open_ab = 'a')
                else:
                    client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
                    print('Train loss is {:.5f}'.format(train_loss), flush=True)
                    central_node = Server_update(args, central_node, client_nodes, select_list, size_weights,noise_multiplier, open_ab = 'ab')
            elif args.method == 'FFA-LoRA':
                client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
                print('Train loss is {:.5f}'.format(train_loss), flush=True)
                central_node = Server_update(args, central_node, client_nodes, select_list, size_weights,noise_multiplier, open_ab = 'b')
            elif args.method == 'LoRA' or args.method == 'DP-DyLoRA':
                client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
                print('Train loss is {:.5f}'.format(train_loss), flush=True)
                central_node = Server_update(args, central_node, client_nodes, select_list, size_weights,noise_multiplier, open_ab = 'ab')
            else:
                assert False
            val_acc, val_recall, val_prec, val_f1, val_auc  = validate(args, central_node, which_dataset = 'validate')
            print('Val acc: {:.3f}'.format(val_acc)+ ', recall: {:.3f}'.format(val_recall)+ ', prec: {:.3f}'.format(val_prec)+ ', f1: {:.3f}'.format(val_f1)+ ', auc: {:.3f}'.format(val_auc), flush=True) 
            print('Test acc: {:.3f}'.format(best_test_acc)+ ', recall: {:.3f}'.format(best_test_recall)+ ', prec: {:.3f}'.format(best_test_prec)+ ', f1: {:.3f}'.format(best_test_f1)+ ', auc: {:.3f}'.format(best_test_auc), flush=True)
            print()
            if val_acc+val_recall+val_prec+val_f1+val_auc>best_val_acc:
                best_val_acc = val_acc+val_recall+val_prec+val_f1+val_auc
                best_test_acc,best_test_recall, best_test_prec, best_test_f1,best_test_auc = validate(args, central_node, which_dataset = 'test')
                print('Test acc: {:.3f}'.format(best_test_acc)+ ', recall: {:.3f}'.format(best_test_recall)+ ', prec: {:.3f}'.format(best_test_prec)+ ', f1: {:.3f}'.format(best_test_f1)+ ', auc: {:.3f}'.format(best_test_auc), flush=True)
                print()
                #torch.save(central_node.model.state_dict(), os.path.join(root_path, output_path, setting_name+'_finalmodel.pth'))
        all_acc.append(best_test_acc)
        all_recall.append(best_test_recall)
        all_prec.append(best_test_prec)
        all_f1.append(best_test_f1)
        all_auc.append(best_test_auc)
        end = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The ending time ：{}'.format(end))
    print('===========================================================')
    print('Best test acc:', all_acc)
    print('Best test acc mean: {:.5f}'.format(np.mean(all_acc)),'Best test acc std: {:.5f}'.format(np.std(all_acc)) )

    print('Best test recall:', all_recall)
    print('Best test recall mean: {:.5f}'.format(np.mean(all_recall)),'Best test recall std: {:.5f}'.format(np.std(all_recall)) )

    print('Best test prec:', all_prec)
    print('Best test prec mean: {:.5f}'.format(np.mean(all_prec)),'Best test prec std: {:.5f}'.format(np.std(all_prec)) )

    print('Best test f1:', all_f1)
    print('Best test f1 mean: {:.5f}'.format(np.mean(all_f1)),'Best test f1 std: {:.5f}'.format(np.std(all_f1)) )

    print('Best test auc:', all_auc)
    print('Best test auc mean: {:.5f}'.format(np.mean(all_auc)),'Best test auc std: {:.5f}'.format(np.std(all_auc)) )

    print('===========================================================')