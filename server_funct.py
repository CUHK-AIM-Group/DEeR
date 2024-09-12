import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.backends import cudnn
from random import sample
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import TensorDataset

##############################################################################
# General server function
##############################################################################

def Server_update(args, central_node, client_nodes, select_list, size_weights, noise_multiplier=None, open_ab = 'ab'):

    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]
    
    # update the global model
    if args.is_DP:
        for key in central_node.model.state_dict().keys():
            if open_ab == 'a':
                condition =  'linear_a' in key
            elif open_ab == 'b':
                condition =  'linear_b' in key
            elif open_ab == 'ab':
                condition =  'linear_a' in key or 'linear_b' in key
            else:
                assert False
            if condition:
                if args.method=='DEeR' and args.module2 == 1:
                    temp = torch.zeros_like(central_node.model.state_dict()[key])
                    for weight_idx, client_idx in enumerate(select_list):
                        if 'linear_a' in key:
                            peer_key = key.replace('linear_a', 'linear_b')
                            dim = temp.shape[1]
                        elif 'linear_b' in key:
                            peer_key = key.replace('linear_b', 'linear_a') # d r r d
                            dim = temp.shape[0]
                        else:
                            assert False
                        nabala = copy.deepcopy(client_nodes[client_idx].model.state_dict()[key] - central_node.model.state_dict()[key])
                        # clip
                        nabala *= min(1, args.C/torch.norm(nabala, 2))
                        # add DP noise
                        w_noise = torch.FloatTensor(dim, dim).normal_(0, noise_multiplier * args.C/np.sqrt(args.node_num)).cuda()
                        peer_weight = client_nodes[client_idx].model.state_dict()[peer_key]
                        if 'linear_a' in key:
                            noise = torch.inverse(peer_weight.T @ peer_weight) @ peer_weight.T @ w_noise
                        elif 'linear_b' in key:
                            noise = w_noise @ peer_weight.T @ torch.inverse(peer_weight @ peer_weight.T)
                        else:
                            assert False
                        nabala = nabala.add_(noise)
                        # aggregate
                        temp += agg_weights[weight_idx] * (central_node.model.state_dict()[key] + nabala)
                    central_node.model.state_dict()[key].data.copy_(temp)
                else:#FedLoRA, FFA-LORA, DP-DyLoRA
                    temp = torch.zeros_like(central_node.model.state_dict()[key])
                    for weight_idx, client_idx in enumerate(select_list):
                        nabala = copy.deepcopy(client_nodes[client_idx].model.state_dict()[key] - central_node.model.state_dict()[key])
                        # clip
                        nabala *= min(1, args.C/torch.norm(nabala, 2))
                        # add DP noise
                        noise = torch.FloatTensor(nabala.shape).normal_(0, noise_multiplier * args.C/np.sqrt(args.node_num)).cuda()
                        nabala = nabala.add_(noise)
                        # aggregate
                        temp += agg_weights[weight_idx] * (central_node.model.state_dict()[key] + nabala)
                    central_node.model.state_dict()[key].data.copy_(temp)
    else:
        for key in central_node.model.state_dict().keys():
            if open_ab == 'a':
                condition =  'linear_a' in key
            elif open_ab == 'b':
                condition =  'linear_b' in key
            elif open_ab == 'ab':
                condition =  'linear_a' in key or 'linear_b' in key
            else:
                assert False
            if condition:
                temp = torch.zeros_like(central_node.model.state_dict()[key])
                for weight_idx, client_idx in enumerate(select_list):
                    temp += agg_weights[weight_idx] * client_nodes[client_idx].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)

    return central_node
