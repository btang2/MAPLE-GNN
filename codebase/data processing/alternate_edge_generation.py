#generate alternate edge_list for all the proteins (default ang_cutoff = 6.0, but we try 7.0, 8.0, 9.0, and 10.0 as well)

import requests
from time import process_time
import os
import modules
import math
import torch
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))

from hfxgnn.metrics import *
from hfxgnn.models import GAT_baseline
from hfxgnn.dataprep import generate_split
from hfxgnn.train import train_model


    
with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_list = []

for data in data_list:
    pdb_list.append(data.strip().split('\t')[1].lower())
#print(pdb_list)
#os.makedirs('pdb_files/', exist_ok=True)
#torch.cuda.empty_cache()
ctr = 0
starttime = process_time()
#pdb_errors = ['4ui9', '5oqm', '5mps', '4v7o', '5a9q', '5luq']

edge_errors = []
num_errors = 0
start_index = 110
for i in range(len(pdb_list)):
    #if (pdb_list[i] == '4ui9'):
    #    print("reached 4ui9")
    if os.path.exists("codebase/data/npy/" + str(pdb_list[i]) + "-edge_list_8_5.npy"):
        #print("alr processed " + str(i))
        continue
    try:
        modules.save_alternate_edges(pdb_list[i]) #add 6, 7, 8, 9, 10 because I messed up the file save...
        modules.save_alternate_edges2(pdb_list[i]) #add 6.5, 7.5, 8.5, 9.5 &will take a while
    except Exception as e:    
        print("!! PDB PROCESSING ERROR: index " + str(i) + ", id = " + str(pdb_list[i]) + " -- " + str(e))
        num_errors += 1
        edge_errors.append(pdb_list[i])
        #pdb_errors.append(pdb_list[i])

    if i%100 == 0:
        cur_time = process_time() - starttime #secs
        print("############ PROCESSED " + str(i) + " OF " + str(len(pdb_list)) + " PDBS")
        print("############ TOTAL TIME ELAPSED: " + str(math.floor(cur_time / (60.0*60.0))) + " hr " + str(math.floor((cur_time % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(cur_time % 60.0)) + " sec")
        completion = cur_time / (i+1 - start_index) * len(pdb_list)
        print("############ TIME REMAINING: " + str(math.floor(completion / (60.0*60.0))) + " hr " + str(math.floor((completion % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(completion % 60.0)) + " sec")
        remain = completion - cur_time
        print("############ ESTIMATED TIME TO COMPLETION: " + str(math.floor(remain / (60.0*60.0))) + " hr " + str(math.floor((remain % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(remain % 60.0)) + " sec")
    #if i%100 == 0:
    #    time.sleep(60)
#print("Errors: " + str(pdb_errors))
print("Error List: " + str(edge_errors))


print("training edge_list models here bc why not")
cutoffs = ["6", "6_5", "7", "7_5", "8", "8_5", "9", "9_5", "10"]
for cutoff in cutoffs:
    dataset, trainloader, testloader = generate_split(cutoff)
    model = GAT_baseline
    modelname = "GAT_baseline_" + cutoff
    train_model(10, model, modelname, trainloader, testloader)

