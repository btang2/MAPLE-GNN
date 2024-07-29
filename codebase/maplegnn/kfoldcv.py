
import os
import torch
import numpy as np 
from sklearn.model_selection import KFold

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from train import train_model
from maplegnn.test import test_model
from models import MH_GATv2_sagpool_GraphConv, MAPLEGNN

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

data_dir = 'codebase/data/npy/'
pairs_file = 'interactions_data.txt' 
#interactions_data.txt for balalnced
#unbalanced_interactions_data.txt for unbalanced

pairs_labels = []

with open(pairs_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        pairs_labels.append((parts[0].lower(), parts[1].lower(), int(parts[2])))  # Convert to lowercase



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


cutoff = "9" 
batch_size = 8
model = MAPLEGNN
modelname = "CROSSVALMODEL"
dataset = GraphPairDataset(pairs_labels, data_dir, cutoff=cutoff, batch_size=batch_size)
print("##### BEGIN CROSS VALIDATION FOR " + pairs_file)
size = len(pairs_labels)
print(size)

kfold = KFold(n_splits = 5, shuffle=True, random_state=1)
kfold.get_n_splits(dataset)
print(kfold)
for i, (train_index, test_index) in enumerate(kfold.split(dataset)):
    print(f"Now Training Fold {i}:")
    trainloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_index), drop_last = True)
    testloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_index))
    foldmodelname = modelname + "-" + str(i)
    print("Training Model " + foldmodelname + "...")
    train_model(50, model, foldmodelname, trainloader, testloader)
    print("Testing Model " + foldmodelname + "...")
    test_model(model, foldmodelname, testloader)