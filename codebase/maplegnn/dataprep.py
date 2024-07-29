import os
import torch
import numpy as np 
import random
import math

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

#create dataloader/train-test split for ablation experiments

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

data_dir = 'codebase/data/npy/'
# Path to the .txt file with pairs and labels
#interactions_data.txt for 1:1 dataset, unbalanced_interactions_data.txt for 1:10 dataset
pairs_file = 'interactions_data.txt'

def generate_unbalanced_dataset_txt():
    #generate a 1:10 PPI split on the dataset by randomly removing positive pdb pairs and saving the interactions into a new text file
    #stored in unbalanced_interactions_data.txt
    pos_pairs = []
    neg_pairs = []
    with open('interactions_data.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if (int(parts[2]) == 1):
                pos_pairs.append((parts[0], parts[1], parts[2]))
            if (int(parts[2]) == 0):
                neg_pairs.append((parts[0], parts[1], parts[2]))
    num_pos_pairs = len(neg_pairs) // 10
    random.shuffle(pos_pairs)
    new_pos_pairs = pos_pairs[:num_pos_pairs]
    combined_pairs = []
    for pos_pair in new_pos_pairs:
        combined_pairs.append(pos_pair)
    for neg_pair in neg_pairs:
        combined_pairs.append(neg_pair)
    random.shuffle(combined_pairs)
    #print(str(len(combined_pairs)))
    pos_count = 0
    neg_count = 0
    with open('unbalanced_interactions_data.txt', 'w') as f:
        for count, (p1, p2, label) in enumerate(combined_pairs):
            if (int(label) == 1):
                pos_count += 1
            if (int(label) == 0):
                neg_count += 1
            f.write(p1 + "\t" + p2 + "\t" + label + "\n")
    print(f"generated 1:10 split dataset, positive samples: {pos_count}, negative samples: {neg_count}")

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


#create dataloader
def generate_split(cutoff, batch_size):
    dataset = GraphPairDataset(pairs_labels, data_dir, cutoff=cutoff, batch_size=batch_size)
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    seed = 42 #PPIGNN used 42
    torch.manual_seed(seed)
    size = len(pairs_labels)
    print(size)
    #print(math.floor(0.8 * size))
    #Make iterables using dataloader class  
    trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size)]) #ablation split
    trainloader = DataLoader(dataset= trainset, batch_size= batch_size, num_workers = 0, collate_fn = collate_fn, drop_last=True) #,  collate_fn = collate_fn
    testloader = DataLoader(dataset= testset, batch_size= batch_size, num_workers = 0, collate_fn = collate_fn, drop_last=True) #regularization of batches
    print("Length")
    #print(len(aug_trainloader))
    print(len(trainset))
    print(len(testset))
    return trainloader, testloader

def filter_test_set(train_pairs, test_pairs):
    train_proteins = set()
    for id1, id2, _ in train_pairs:
        train_proteins.add(id1)
        train_proteins.add(id2)

    filtered_test_pairs = [pair for pair in test_pairs if pair[0] not in train_proteins and pair[1] not in train_proteins]
    return filtered_test_pairs

def generate_strict_split(cutoff, batch_size, test_size, balanced):
    pairs_labels = []
    if (balanced):
        pairs_file = "interactions_data.txt"
    else:
        pairs_file = "unbalanced_interactions_data.txt"
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            pairs_labels.append((parts[0].lower(), parts[1].lower(), int(parts[2])))  # Convert to lowercase
    train_pairs, test_pairs = train_test_split(pairs_labels, test_size=test_size, random_state=42)
    test_pairs = filter_test_set(train_pairs, test_pairs)
    
    train_pairs_neg = 0
    train_pairs_pos = 0
    test_pairs_neg = 0
    test_pairs_pos = 0
    for pair in train_pairs:
        if (pair[2] == 1):
            train_pairs_pos += 1
        else:
            train_pairs_neg += 1
    for pair in test_pairs:
        if (pair[2] == 1):
            test_pairs_pos += 1
        else:
            test_pairs_neg += 1
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"train: {str(len(train_pairs))} ({str(train_pairs_pos)} pos, {str(train_pairs_neg)} neg), test: {str(len(test_pairs))} ({str(test_pairs_pos)} pos, {str(test_pairs_neg)} neg)")

    train_dataset = GraphPairDataset(train_pairs, data_dir, cutoff=cutoff, batch_size=batch_size)
    test_dataset = GraphPairDataset(test_pairs, data_dir, cutoff=cutoff, batch_size=batch_size)
    trainloader = DataLoader(dataset= train_dataset, batch_size= batch_size, num_workers = 0, collate_fn = collate_fn, drop_last=True) #,  collate_fn = collate_fn
    testloader = DataLoader(dataset= test_dataset, batch_size= batch_size, num_workers = 0, collate_fn = collate_fn) #regularization of batches
    
    return  trainloader, testloader

