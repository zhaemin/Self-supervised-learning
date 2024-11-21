import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.nn.utils as utils

import dataloader

from utils import parsing_argument, load_model, set_parameters

from torch.utils.tensorboard import SummaryWriter


def train_per_epoch(args, dataloader, net, optimizer, device):
    running_loss = 0.0
    net.train()
    
    representations = None
    label_list = None
    
    for data in dataloader:
        inputs, labels = data
        
        inputs = torch.stack(inputs)
        inputs, labels = inputs.to(device), labels.to(device)
        loss = net(inputs, device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if representations == None:
                representations = net.encoding(inputs[2])
                label_list = labels
            else:
                representations = torch.cat((representations, net.encoding(inputs[2])), dim=0)
                label_list = torch.cat((label_list, labels), dim=0)
            
        running_loss += loss.item()
        
    return running_loss/len(dataloader), representations, label_list

def knn_test(testloader, net, representations, label_list, device):
    total = 0
    correct = 0
    net.eval()
    
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            x = net.encoding(inputs)
            distances = torch.cdist(x, representations)
            min_indices = torch.argmin(distances, dim=-1)
            predictions = torch.tensor([label_list[idx].item() for idx in min_indices]).to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    acc = 100 * correct / total
    
    return acc

def train(args, trainloader, testloader, net, optimizer, scheduler, device, writer, outputs_log):
    
    for epoch in range(args.epochs):
        running_loss, representations, label_list = train_per_epoch(args, trainloader, net, optimizer, device)
        acc = knn_test(testloader, net, representations, label_list, device)
        lr = optimizer.param_groups[0]['lr']
        
        print('epoch[%d] - training loss : %.3f / KNN_acc : %.3f / lr : %f'%(epoch+1, running_loss, acc, lr))
        print('epoch[%d] - training loss : %.3f / KNN_acc : %.3f'%(epoch+1, running_loss, acc), file=outputs_log)
        writer.add_scalar('train / train_loss', running_loss, epoch+1)
        writer.add_scalar('train / KNN_acc', acc, epoch+1)
        writer.add_scalar('train / learning_rate', lr, epoch+1)
        
        torch.save(net.state_dict(), './moco.pt')
        
        running_loss = 0.0
        
        if scheduler:
            scheduler.step()
        
    print('Training finished',file=outputs_log)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parsing_argument()
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    outputs_log = open(f'./outputs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{cur_time}.txt','w')
    writer = SummaryWriter(f'./logs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{cur_time}')
    
    trainloader,testloader, num_classes = dataloader.load_dataset(args)
    
    net = load_model(args)
    net.to(device)
    #net.load_state_dict(torch.load('./moco_pretext.pt'))
    optimizer,scheduler = set_parameters(args, net)
    
    if args.train:
        print(f"Training start ...")
        train(args, trainloader, testloader, net, optimizer, scheduler, device, writer, outputs_log)
        
    outputs_log.close()
    writer.close()

if __name__ == "__main__":
    main()
