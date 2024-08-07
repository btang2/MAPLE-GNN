import torch
import numpy as np
import matplotlib.pyplot as plt
from metrics import get_cross_entropy, get_accuracy, precision, sensitivity, specificity, f_score, mcc, auroc, auprc
from metrics import get_tp, get_fp, get_tn, get_fn
from dataprep import generate_split
from models import GAT_plm, GAT_plm_dssp, GAT_plm_edgefeat, GAT_plm_dssp_edgefeat, GAT_plm_dssp_edgefeat_sagpool
from models import GAT_sagpool_baseline, GCN_sagpool, GATv2_sagpool, MH_GATv2_sagpool, GT_sagpool, MH_GT_sagpool
from models import MH_GATv2_sagpool_GraphConv, MH_GATv2_sagpool_GCNConv, MH_GATv2_sagpool_GATConv, MH_GATv2_sagpool_GATv2Conv, MH_GATv2_sagpool_SAGEConv
from models import MAPLEGNN

def test_model(model, modelname, testloader):
  print("##### TESTING RESULTS for " + str(modelname))
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
  #print(device)
  model = model()
  modelpath = "codebase/model_instances/" + str(modelname) + ".pth" #path to load the model
  model.load_state_dict(torch.load(modelpath))
  model.to(device)
  model.eval()
  predictions = torch.Tensor()
  labels = torch.Tensor()
  with torch.no_grad():
      for prot_1, prot_2, label in testloader:
        prot_1 = prot_1.to(device)
        prot_2 = prot_2.to(device)
        output = model(prot_1, prot_2)
        predictions = torch.cat((predictions, output.cpu()), 0)
        labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
  labels = labels.numpy().flatten()
  predictions = predictions.numpy().flatten()

  tn = get_tn(labels, predictions, 0.5)
  tp = get_tp(labels, predictions, 0.5)
  fn = get_fn(labels, predictions, 0.5)
  fp = get_fp(labels, predictions, 0.5)

  loss_ = get_cross_entropy(labels, predictions)
  acc_ = get_accuracy(labels, predictions, 0.5)
  prec_ = precision(labels, predictions, 0.5)
  sensitivity_ = sensitivity(labels, predictions,  0.5)
  specificity_ = specificity(labels, predictions, 0.5)
  f1_ = f_score(labels, predictions, 0.5)
  mcc_ = mcc(labels, predictions,  0.5)
  auroc_ = auroc(labels, predictions)
  auprc_ = auprc(labels, predictions)


  print(f'loss : {loss_}')
  print(f'Accuracy : {acc_}')
  print(f'precision: {prec_}')
  print(f'Sensititvity :{sensitivity_}')
  print(f'specificity : {specificity_}')
  print(f'f-score : {f1_}')
  print(f'MCC : {mcc_}')
  print(f'AUROC: {auroc_}')
  print(f'AUPRC: {auprc_}')
  print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}') #this should be ok

#trainloader, testloader = generate_split("9", 8)
#model = MH_GATv2_sagpool_GraphConv
#modelname = "CROSSVALMODEL-0"
#test_model(model, modelname, testloader)

#cutoffs = ["6", "6_5", "7", "7_5", "8", "8_5", "9", "9_5", "10"]
#for cutoff in cutoffs:
#    trainloader, testloader = generate_split(cutoff, 8)
#    model = MH_GATv2_sagpool_GraphConv
#    modelname = "MH-GATv2-sagpool-" + str(cutoff) 
#    test_model(model, modelname, testloader)

#models = [GCN_sagpool, GAT_plm_dssp_edgefeat_sagpool, GATv2_sagpool, MH_GATv2_sagpool, GT_sagpool, MH_GT_sagpool]
#modelnames = ["GCN-sagpool", "GAT-sagpool", "GATv2-sagpool", "MH-GATv2-sagpool", "GT-sagpool", "MH-GT-sagpool"]
#trainloader, testloader = generate_split("9", 8)
#for i in range(len(models)):
#  model = models[i]
#  modelname = modelnames[i]
#  test_model(model, modelname, testloader)
#trainloader, testloader = generate_strict_split("9", 8, 0.98, balanced=True)
#test_model(model, modelname, testloader)