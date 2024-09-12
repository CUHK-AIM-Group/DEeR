

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
# from Data import DatasetSplit
from datasets import DatasetSplit
from utils import init_model
from utils import init_optimizer
from opacus.accountants.utils import get_noise_multiplier
class Node(object):

    def __init__(self,args, num_id, train_loader, val_loader, test_loader):
        self.num_id = num_id
        self.args = args
        self.node_num = self.args.node_num

        if self.args.dataset == 'OCT':
            self.num_classes = 8
        elif self.args.dataset == 'Kvasir':
            self.num_classes = 8
            
        self.local_data = None
        self.validate_set = None
        self.test_set = None

        if args.iid == 1 or num_id == -1:
            self.validate_set, self.test_set = val_loader, test_loader
        else:
            self.local_data = train_loader
            self.sample_per_class = self.generate_sample_per_class(self.local_data)
        
        self.model, self.text_features = init_model(num_id, self.args.local_model, self.args)
        self.model = self.model.cuda()
        if len(self.text_features)>0:
            self.text_features = self.text_features.cuda()
        self.optimizer = init_optimizer(self.num_id, self.model, args)


    def train_val_split(self, idxs, train_set, valid_ratio): 

        np.random.shuffle(idxs)

        validate_size = valid_ratio * len(idxs)
        # print(len(idxs))

        idxs_test = idxs[:int(validate_size)]
        idxs_train = idxs[int(validate_size):]
        print(len(idxs_train),len(idxs_test))

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True)

        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize,  num_workers=0, shuffle=True)
        

        return train_loader, test_loader

    def train_val_split_forServer(self, idxs, train_set, valid_ratio, num_classes=10): # local data index, trainset

        np.random.shuffle(idxs)

        validate_size = valid_ratio * len(idxs)

        # generate proxy dataset with balanced classes
        idxs_test = []
        test_class_count = [int(validate_size)/num_classes for _ in range(num_classes)]
        k = 0
        while sum(test_class_count) != 0:
            if test_class_count[train_set[idxs[k]][1]] > 0:
                idxs_test.append(idxs[k])
                test_class_count[train_set[idxs[k]][1]] -= 1
            else: 
                pass
            k += 1
        label_list = []
        for k in idxs_test:
            label_list.append(train_set[k][1])

        idxs_train = [idx for idx in idxs if idx not in idxs_test]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True)
        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize,  num_workers=0, shuffle=True)

        return train_loader, test_loader

    def generate_sample_per_class(self, local_data):
        sample_per_class = torch.tensor([0 for _ in range(self.num_classes)])

        for idx, (data, target) in enumerate(local_data):
            sample_per_class += torch.tensor([sum(target==i) for i in range(self.num_classes)])

        sample_per_class = torch.where(sample_per_class > 0, sample_per_class, 1)

        return sample_per_class

    def compute_sum_proto_cos(self):
        train_loader = self.local_data  # iid
        cos_per_label = [[] for _ in range(self.num_classes)]
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                proto = self.model.proto_classifier(target)
                feature, _, _ = self.model(data)
                proto_cos = torch.bmm(feature.unsqueeze(1), proto.unsqueeze(2)).view(-1) 

                for i, label in enumerate(target):
                    cos_per_label[label].append(proto_cos[i])

        cos_per_label = [sum(item)/len(item) if item != [] else 0 for item in cos_per_label]
        cos_per_label = torch.tensor(cos_per_label)

        return cos_per_label.sum()