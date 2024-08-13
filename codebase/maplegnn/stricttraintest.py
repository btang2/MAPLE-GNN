import os
import torch
import numpy as np 
import random
import math
import sys
from datetime import datetime
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from maplegnn.train import train_model
from maplegnn.test import test_model
from maplegnn.models import MAPLEGNN, MAPLEGNN_exp

#create dataloader/train-test split for ablation experiments

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

data_dir = 'codebase/data/npy/'
# Path to the .txt file with pairs and labels
#interactions_data.txt for 1:1 dataset, unbalanced_interactions_data.txt for 1:10 dataset

# Custom Dataset to load graph data on the fly
class GraphPairDataset(Dataset):
    def __init__(self, pairs_labels, data_dir, cutoff, batch_size):
        self.pairs_labels = pairs_labels
        self.data_dir = data_dir
        self.cutoff = cutoff
        self.batch_size = batch_size

    def __len__(self):
        return len(self.pairs_labels)

    def __getitem__(self, idx):
        id1, id2, label = self.pairs_labels[idx]
        graph1 = self.load_graph_data(id1)
        graph2 = self.load_graph_data(id2)
        return graph1, graph2, label

    def load_graph_data(self, graph_id):
        node_features = np.load(os.path.join(self.data_dir, f'{graph_id}-node_feat_reduced_dssp.npy')) #normalized or reduced depending on model
        edge_indices = np.load(os.path.join(self.data_dir, f'{graph_id}-edge_list_{self.cutoff}.npy')) 
        edge_features = np.load(os.path.join(self.data_dir, f'{graph_id}-edge_feat_{self.cutoff}.npy'))

        node_features = torch.from_numpy(node_features).float().to(device)
        edge_indices = torch.from_numpy(edge_indices).long().t().T.contiguous().to(device) #why need to T?
        edge_features = torch.from_numpy(edge_features).float().to(device)

        return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features, batch = self.batch_size, graph_id = graph_id)

# Custom collate function for DataLoader
def collate_fn(batch):
    graph1_batch, graph2_batch, label_batch = zip(*batch)
    return list(graph1_batch), list(graph2_batch), torch.tensor(label_batch, dtype=torch.float32)

def load_data(cutoff, batch_size, file_suffix=""):
    train_file = 'strict_splits/train_interactions_data' + file_suffix + '.txt'
    test_file = 'strict_splits/test_interactions_data' + file_suffix + '.txt'
    train_pairs_labels = []
    test_pairs_labels = []
    with open(train_file, 'r') as f1:
        for line1 in f1:
            parts1 = line1.strip().split()
            train_pairs_labels.append((parts1[0].lower(), parts1[1].lower(), int(parts1[2])))  # Convert to lowercase
    with open(test_file, 'r') as f2:
        for line2 in f2:
            parts2 = line2.strip().split()
            test_pairs_labels.append((parts2[0].lower(), parts2[1].lower(), int(parts2[2])))  # Convert to lowercase
    train_dataset = GraphPairDataset(train_pairs_labels, data_dir, cutoff=cutoff, batch_size=batch_size)
    test_dataset = GraphPairDataset(test_pairs_labels, data_dir, cutoff=cutoff, batch_size=batch_size)
    seed = 3407
    torch.manual_seed(seed)
    print(f"dataset size: train: {len(train_pairs_labels)}, test: {len(test_pairs_labels)}")
    trainloader = DataLoader(dataset = train_dataset, batch_size = batch_size, num_workers = 0, collate_fn = collate_fn, shuffle=True)
    testloader = DataLoader(dataset = test_dataset, batch_size =  batch_size, num_workers = 0, collate_fn = collate_fn)
    return train_dataset, test_dataset, trainloader, testloader





cutoff = "9" #9
batchsize = 8
model = MAPLEGNN

for i in range(10):
    modelname = "MAPLEGNN-VAL-" + str(i)
    print("!!!!!! NOW TRAINING AND TESTING: " + modelname)

    train_dataset, test_dataset, trainloader, testloader = load_data(cutoff, batchsize, str(i))
    start_time = datetime.now()
    train_model(20, model, modelname, trainloader, testloader) #20 epochs, could lower for lower traintime or raise to see growth
    end_time = datetime.now()
    time_diff = (end_time - start_time).total_seconds()
    print(modelname + " training time: " + str(time_diff))
    test_model(model, modelname, testloader)

#hopefully good
