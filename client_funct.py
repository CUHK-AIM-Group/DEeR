from cProfile import label
import numpy as np
import torch
import torch.nn.functional as F
from utils import validate
import copy
from nodes import Node

##############################################################################
# General client function 
##############################################################################

def receive_server_model(args, client_nodes, central_node):

    for idx in range(len(client_nodes)):
        client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

    return client_nodes

def Client_update(args, client_nodes, central_node, select_list):
    '''
    client update functions
    '''
    # clients receive the server model 
    client_nodes = receive_server_model(args, client_nodes, central_node)

    # update the global model
    client_losses = []
    for i in select_list:
        epoch_losses = []
        for epoch in range(args.E):
            if args.method == 'DP-DyLoRA':
                new_rank = torch.randint(0,args.lora_r,(1,)).item()
                client_nodes[i].model.set_rank(new_rank, frozen=False)
            loss = client_localTrain(args, client_nodes[i])
            epoch_losses.append(loss)
        client_losses.append(sum(epoch_losses)/len(epoch_losses))
        train_loss = sum(client_losses)/len(client_losses)

    return client_nodes, train_loss


def client_localTrain(args, node, loss = 0.0):
    torch.cuda.empty_cache()  # 释放显存
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data, target = data.cuda(), target.cuda()
        image_features = node.model(data)
        text_features = node.text_features
        logit_scale = node.model.logit_scale.exp()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        output_local = logit_scale * image_features @ text_features.T # B 512   C 512

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss += loss_local.item()
        node.optimizer.step()
        
        del loss_local

    return loss/len(train_loader)

