import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class SSLTransform(torch.nn.Module):
    def __init__(self):
        super(SSLTransform, self).__init__()
        self.transform = transforms.Compose([ 
            transforms.RandomResizedCrop((32,32)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        self.transform_test = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        x = self.transform_test(x)
        
        return [x1, x2, x]
    
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
        trainset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=False, download=True, transform=transform_test)
        num_classes = 100
        
    ssltransform = SSLTransform()
    trainset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=True, download=True, transform=ssltransform)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
    return trainloader,testloader,num_classes