from time import process_time
import os
import modules
import math
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.insert(0, str(path_root))

from maplegnn.metrics import *
from maplegnn.models import GAT_baseline
from maplegnn.dataprep import generate_split
from maplegnn.train import train_model

#generate edge lists and edge features with alternate distance cutoffs
    
with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_list = []

for data in data_list:
    pdb_list.append(data.strip().split('\t')[1].lower())

ctr = 0
starttime = process_time()
edge_errors = []
num_errors = 0
start_index = 110
for i in range(len(pdb_list)):
    if os.path.exists("codebase/data/npy/" + str(pdb_list[i]) + "-edge_list_9.npy"):
        print("ALREADY PROCESSED " + str(i))
        continue
    try:
        modules.save_alternate_edges(pdb_list[i]) #add 6, 7, 8, 9, 10
        modules.save_alternate_edges2(pdb_list[i]) #add 6.5, 7.5, 8.5, 9.5
    except Exception as e:    
        print("!! PDB PROCESSING ERROR: index " + str(i) + ", id = " + str(pdb_list[i]) + " -- " + str(e))
        num_errors += 1
        edge_errors.append(pdb_list[i])

    if i%100 == 0:
        cur_time = process_time() - starttime #secs
        print("############ PROCESSED " + str(i) + " OF " + str(len(pdb_list)) + " PDBS")
        print("############ TOTAL TIME ELAPSED: " + str(math.floor(cur_time / (60.0*60.0))) + " hr " + str(math.floor((cur_time % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(cur_time % 60.0)) + " sec")
        completion = cur_time / (i+1 - start_index) * len(pdb_list)
        print("############ TIME REMAINING: " + str(math.floor(completion / (60.0*60.0))) + " hr " + str(math.floor((completion % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(completion % 60.0)) + " sec")
        remain = completion - cur_time
        print("############ ESTIMATED TIME TO COMPLETION: " + str(math.floor(remain / (60.0*60.0))) + " hr " + str(math.floor((remain % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(remain % 60.0)) + " sec")
    
print("Error List: " + str(edge_errors))