from email.policy import default
from typing_extensions import Required
from data import dataloader
from config import args
from utils import transform, plot
import train
from argparse import ArgumentParser
import models, train

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

data= dataloader.CustomDataSet(args.metadata, args.path, transform= transform)
batch_size= args.bs
train_size= args.train_size
train_size= int(train_size*len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Model
parser= ArgumentParser()
parser.add_argument('-m', '--model', help= 'chose the model', default='resnet', required= True)
parser.add_argument('-n', '--num_epoch', help= 'give the numbe of epochs', default=1e-4, required= True, type= int)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
main_args= vars(parser.parse_args())

#
if main_args['model'].lower()=='resnet':
    model= models.resnet50()
elif main_args['model'].lower()=='cnn':
    model = models.CNN().to(device)

criterion= nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=1.0)
num_epochs= main_args['num_epoch']

model_trained, percent, val_loss, val_acc, train_loss, train_acc= train.train(model, criterion, train_loader, val_loader, optimizer, num_epochs, device)
plot(train_loss, val_loss)