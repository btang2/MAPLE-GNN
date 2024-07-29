import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math
import sklearn
import torch_optimizer as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR
from metrics import *
import time

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from dataprep import generate_split, generate_strict_split
from models import GAT_plm, GAT_plm_edgefeat, GAT_plm_dssp, GAT_plm_dssp_edgefeat, GAT_plm_dssp_edgefeat_sagpool, GAT_plm_1dmf_dssp_edgefeat_sagpool
from models import GAT_sagpool_baseline, GCN_sagpool, GATv2_sagpool, MH_GATv2_sagpool, GT_sagpool, MH_GT_sagpool
from models import MH_GATv2_concat_sagpool_GraphConv, MH_GATv2_sagpool_GraphConv, MH_GATv2_sagpool_GCNConv, MH_GATv2_sagpool_GATConv, MH_GATv2_sagpool_GATv2Conv, MH_GATv2_sagpool_SAGEConv, MH_GATv2_sagpool_sepGraphConv
from models import MAPLEGNN
from torch_geometric.loader import DataLoader
import random


def train_model(epochs, model, modelname, trainloader, testloader):
  print("Datalength")
  print(len(trainloader))
  print(len(testloader))
  
  #for plotting
  train_loss = []
  val_loss = []
  train_acc = []
  val_acc = []
  
  def train(model, device, trainloader, optimizer, epoch):
    #random_indices = torch.randperm(len(trainloader))[:800] #make sure to disable/enable when needed
    #random_indices = torch.from_numpy(np.arange(len(trainloader))) #disable random indices 
    #random_indices = torch.sort(random_indices)[0]
    #print(f'Training on {len(random_indices)} random samples chosen out of {len(trainloader)} training samples.....')
    #print(f'Training on {len(trainloader)} training samples.....')
    #print(random_indices)
    model.train()
    loss_func = nn.BCELoss() #cross entropy instead of MSE
    predictions_tr = torch.Tensor().to(device)
    scheduler = MultiStepLR(optimizer=optimizer, milestones = [10, 20, 24, 28, 32, 36, 40], gamma=0.5) #38
    labels_tr = torch.Tensor().to(device)
    #aug_predictions_tr = torch.Tensor().to(device)
    #index = 0
    for count,(prot_1, prot_2, label) in enumerate(trainloader):
      #if (index >= len(random_indices)):
      #  break
      #if (random_indices[index] == count):
      if (count % 500 == 0):
        print(f'epoch {epoch}, examples done: {count}')
      
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      optimizer.zero_grad()
      output = model(prot_1, prot_2)

      predictions_tr = torch.cat((predictions_tr, output.to(device)), 0)
      labels_tr = torch.cat((labels_tr, label.view(-1,1).to(device)), 0)
      loss = loss_func(output.to(device), label.view(-1,1).float().to(device))
      loss.backward()
      optimizer.step()
      #index += 1
      
    scheduler.step()
    labels_tr = labels_tr.cpu().detach().numpy()
    predictions_tr = predictions_tr.cpu().detach().numpy()
    L = labels_tr.flatten()
    P = predictions_tr.flatten()
    #print( f'Train Predictions---------------------------------------------{P}')
    print(f'Train Prediction max: {np.max(P)}')
    print(f'Train Prediction min: {np.min(P)}')
    print(f'Train Prediction avg: {np.mean(P)}')
    print(f'Train Prediction std: {np.std(P)}')
    #print(f'Train Labels----------------------------------------------------{L}')
    #print(labels_tr)
    #print(predictions_tr)
    acc_tr = get_accuracy(labels_tr, predictions_tr , 0.5)
    print(f'Epoch {epoch} / {epochs} [==============================] - train_loss : {loss} - train_accuracy : {acc_tr}') #just 15 instead of 30 epochas
    train_acc.append(acc_tr)
    train_loss.append(loss.cpu().detach().numpy())

  def predict(model, device, loader):
    model.eval()
    predictions = torch.Tensor() #on CPU
    labels = torch.Tensor()
    with torch.no_grad():
      for prot_1, prot_2, label in loader:
        prot_1 = prot_1.to(device)
        prot_2 = prot_2.to(device)
        #print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
        output = model(prot_1, prot_2)
        #output_2 = model(prot_2, prot_1) #and then concatenate to predictions & labels for data augmentation
        #print(output[0])
        predictions = torch.cat((predictions, output.cpu()), 0) 
        labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
        
    labels = labels.numpy()
    predictions = predictions.numpy()
    return labels.flatten(), predictions.flatten()

  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
  #starttime = time.process_time()
  model = model() #gatv2 or gatv2 mutual attenntion (shared conv function)
  model.to(device)
  modelpath = "codebase/model_instances/" + str(modelname) + ".pth" #path to save the model
  num_epochs = epochs
  loss_func = nn.BCELoss() 
  min_loss = 100
  best_accuracy = 0
  optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
  for epoch in range(num_epochs):
    train(model, device, trainloader, optimizer, epoch+1) #for debugging
    
    G, P = predict(model, device, testloader)
    #print( f'Predictions---------------------------------------------{P}')
    print(f'Val Prediction max: {np.max(P)}')
    print(f'Val Prediction min: {np.min(P)}')
    print(f'Val Prediction avg: {np.mean(P)}')
    print(f'Val Prediction std: {np.std(P)}')
    #print(f'Labels----------------------------------------------------{G}')
    loss = get_cross_entropy(G,P)
    accuracy = get_accuracy(G,P, 0.5)
    val_loss.append(loss)
    val_acc.append(accuracy)
    print(f'Epoch {epoch+1} / {num_epochs} [==============================] - val_loss : {loss} - val_accuracy : {accuracy}')

    if(accuracy > best_accuracy):
      best_accuracy = accuracy
      best_acc_epoch = epoch
      torch.save(model.state_dict(), modelpath)
      print("##### New Model Best")
    if(loss< min_loss):
      min_loss = loss
      min_loss_epoch = epoch

  print(f'min_val_loss : {min_loss} for epoch {min_loss_epoch} ............... best_val_accuracy : {best_accuracy} for epoch {best_acc_epoch}')
  print("##### Model saved")
  print("plotting accuracy and loss graphs:")
  
  plt.subplot(1,2,1)
  plt.plot(train_acc, color='b')
  plt.plot(val_acc, color='r')
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.subplot(1,2,2)
  plt.plot(train_loss, color='b')
  plt.plot(val_loss, color='r')
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.tight_layout()
  plt.savefig(f"codebase/images/" + str(modelname) + ".png")
  plt.clf()
  plt.close()


"""
relevant modelnames:
GAT-plm, GAT-plm-edgefeat, GAT-plm-dssp-edgefeat, GAT-plm-dssp-edgefeat (feature implementation ablation)
GCN-sagpool, GAT-sagpool, GATv2-sagpool, MH-GATv2-sagpool, GT-sagpool, MH-GT-sagpool (graph convolution ablation)
MAPLEGNN: Model used for 5-fold CV
"""

