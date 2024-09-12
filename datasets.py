import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import math
import random
import copy
import functools
from PIL import Image
import os
import glob
from torch.utils.data import DataLoader
# from sample_dirichlet import clients_indices

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float() 
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return self.images.shape[0]

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]] 
        return image, label

class CustomSubset(torch.utils.data.Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        dataset.targets = torch.tensor(dataset.targets)
        # print(dataset.targets)
        self.targets = dataset.targets[indices]
        # print(len(self.targets))
        self.classes = dataset.classes 
        self.indices = indices

    def __getitem__(self, idx): 
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self):
        return len(self.indices)

OCT_EXTENSION = 'jpg'
OCT_label_text_to_number = {'AMD':0, 'CNV':1, 'CSR':2, 'DME':3, 'DR':4, 'DRUSEN':5,'MH':6,'NORMAL':7}

class OCT(Dataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(self.root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % OCT_EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        # build class label - number mapping
        for file_path in self.image_paths:
            class_name = file_path.split('/')[-2]
            self.labels[os.path.basename(file_path)] = OCT_label_text_to_number[class_name]

        # get targets
        self.targets = []
        for index in range(len(self.image_paths)):
            file_path = self.image_paths[index]
            label_numeral = self.labels[os.path.basename(file_path)]
            self.targets.append(label_numeral)

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img) if self.transform else img



Kvasir_EXTENSION = 'jpg'
Kvasir_label_text_to_number = {'dyed-lifted-polyps':0, 'dyed-resection-margins':1, 'esophagitis':2, 'normal-cecum':3, 'normal-pylorus':4, 'normal-z-line':5,'polyps':6,'ulcerative-colitis':7}

class Kvasir(Dataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(self.root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % Kvasir_EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        # build class label - number mapping
        for file_path in self.image_paths:
            class_name = file_path.split('/')[-2]
            self.labels[os.path.basename(file_path)] = Kvasir_label_text_to_number[class_name]

        # get targets
        self.targets = []
        for index in range(len(self.image_paths)):
            file_path = self.image_paths[index]
            label_numeral = self.labels[os.path.basename(file_path)]
            self.targets.append(label_numeral)

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img) if self.transform else img



class Data(object):
    def __init__(self, args):
        self.args = args
        node_num = args.node_num
        num_classes = args.num_classes
        if args.dataset == 'OCT':
            # Data enhancement
            tra_transformer = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

            val_transformer = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            self.train_set = OCT(args.data_dir, 'train', tra_transformer, in_memory=False)
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                groups, proportion = build_non_iid_by_dirichlet(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=num_classes, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
                print(proportion)
            else:
                data_num = [int(len(self.train_set)/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.train_loaders = []
            for i in range(node_num):
                print('Client', i, 'Train sample size', len(self.train_loader[i]))
                self.train_loaders.append(DataLoader(DatasetSplit(self.train_set, self.train_loader[i]),
                                      batch_size=self.args.batchsize, num_workers=2, shuffle=True))

            self.val_set = OCT(args.data_dir, 'val', val_transformer, in_memory=False)
            self.val_loader=DataLoader(self.val_set, batch_size=self.args.batchsize, num_workers=2, shuffle=False)
            print('Val sample size', len(self.val_set))
            
            self.test_set = OCT(args.data_dir, 'test', val_transformer, in_memory=False)
            self.test_loader=DataLoader(self.test_set, batch_size=self.args.batchsize, num_workers=2, shuffle=False)
            print('Test sample size', len(self.test_set))


        elif args.dataset == 'Kvasir':
            # Data enhancement
            tra_transformer = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
                
            val_transformer = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            self.train_set = Kvasir(args.data_dir, 'train', tra_transformer, in_memory=False)
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                groups, proportion = build_non_iid_by_dirichlet(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=num_classes, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
                print(proportion)
            else:
                data_num = [int(len(self.train_set)/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.train_loaders = []
            for i in range(node_num):
                print('Client', i, 'Train sample size', len(self.train_loader[i]))
                self.train_loaders.append(DataLoader(DatasetSplit(self.train_set, self.train_loader[i]),
                                      batch_size=self.args.batchsize, num_workers=2, shuffle=True))

            self.val_set = Kvasir(args.data_dir, 'val', val_transformer, in_memory=False)
            self.val_loader=DataLoader(self.val_set, batch_size=self.args.batchsize, num_workers=2, shuffle=False)
            print('Val sample size', len(self.val_set))
            
            self.test_set = Kvasir(args.data_dir, 'test', val_transformer, in_memory=False)
            self.test_loader=DataLoader(self.test_set, batch_size=self.args.batchsize, num_workers=2, shuffle=False)
            print('Test sample size', len(self.test_set))
        else:
            assert 0


def build_non_iid_by_dirichlet(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):
    
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition