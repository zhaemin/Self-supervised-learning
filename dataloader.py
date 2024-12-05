import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Sampler, DataLoader

import numpy as np

class SSLTransform(torch.nn.Module):
    def __init__(self, img_size):
        super(SSLTransform, self).__init__()
        self.transform_strong = transforms.Compose([ 
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.2 ,1)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            #transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        self.transform_weak = transforms.Compose([ 
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.2 ,1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        self.transform_test = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
    def __call__(self, x):
        x1 = self.transform_strong(x)
        x2 = self.transform_weak(x)
        x = self.transform_test(x)
        
        return [x1, x2, x]

class FewShotSampler(Sampler):
    def __init__(self, labels, num_ways, num_shots, num_queries, episodes, num_tasks, data_source=None):
        super().__init__(data_source)
        self.labels = labels
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.episodes = episodes
        self.num_tasks = num_tasks
        
        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.num_classes = len(self.classes)
        self.data_matrix = torch.Tensor(np.empty((len(self.classes), max(self.counts)), dtype=int)*np.nan)
        self.num_per_class = torch.zeros_like(self.classes)
        
        #data_matrix => 해당 class에 맞는 데이터의 index를 저장
        #np.where => nan인 값들이 2차원으로 반환됨 [[nan, nan, ..., nan]]
        
        for data_idx, label in enumerate(labels):
            self.data_matrix[label, np.where(np.isnan(self.data_matrix[label]))[0][0]] = data_idx 
            self.num_per_class[label] += 1
        
        self.valid_classes = [c.item() for c, count in zip(self.classes, self.num_per_class) if count >= self.num_shots+self.num_queries]
        
    def __iter__(self):
        for _ in range(self.episodes):
            tasks = []
            for t in range(self.num_tasks):
                batch_support_set = torch.LongTensor(self.num_ways*self.num_shots)
                batch_query_set = torch.LongTensor(self.num_ways*self.num_queries)
                
                way_indices = torch.randperm(len(self.valid_classes))[:self.num_ways]
                selected_classes = [self.valid_classes[idx] for idx in way_indices]
                
                for i, label in enumerate(selected_classes):
                    slice_for_support = slice(i*self.num_shots, (i+1)*self.num_shots)
                    slice_for_queries = slice(i*self.num_queries, (i+1)*self.num_queries)
                    
                    samples = torch.randperm(self.num_per_class[label])[:self.num_shots+self.num_queries]
                    batch_support_set[slice_for_support] = self.data_matrix[label][samples][:self.num_shots]
                    batch_query_set[slice_for_queries] = self.data_matrix[label][samples][self.num_shots:]
                
                batch = torch.cat((batch_support_set, batch_query_set))
                tasks.append(batch)
            
            batches = torch.cat(tasks)
            yield batches
            
    def __len__(self):
        return self.episodes

def load_dataset(args):
    transform_train = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    if args.dataset == 'cifar10':
        ssltransform = SSLTransform(32)
        trainset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=True, download=True, transform=ssltransform)
        testset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=False, download=True, transform=transform_test)
        
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
        valloader = None
        
        num_classes = 10
    
    elif args.dataset == 'miniimagenet':
        ssltransform = SSLTransform(84)
        trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=ssltransform)
        testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/test', transform=transform_test)
        valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/val', transform=transform_test)
        num_classes = 64
        
        testset_labels = torch.LongTensor(testset.targets)
        valset_labels = torch.LongTensor(valset.targets)
        
        test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 150, num_tasks=4)
        val_sampler = FewShotSampler(valset_labels, args.test_num_ways, args.num_shots, args.num_queries, 25, num_tasks=4)
        
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        testloader = DataLoader(testset, batch_sampler=test_sampler, pin_memory=True)
        valloader = DataLoader(valset, batch_sampler=val_sampler, pin_memory=True)
        
    return trainloader, testloader, valloader, num_classes