import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
from torch_geometric.nn import GraphConv, GCNConv, GATConv, GATv2Conv, TransformerConv, GPSConv, SAGEConv, global_mean_pool as gep, global_max_pool as gmp
from torch_geometric.nn import SAGPooling, ASAPooling
from maplegnn.dataprep import generate_split
from maplegnn.metrics import get_accuracy
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

#Ablation and MAPLE-GNN Models

#Ablation I: Feature Implmementation
class GAT_plm(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024, output_dim=128, dropout=0.2, heads = 1): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GAT w/ plm')
        super(GAT_plm, self).__init__()

        self.hidden = 8 #hyperparameter, though 8 seems to perform best
        self.heads = heads #multi head attention? try performance w/ more heads (3?)

        self.n_output = n_output
        self.conv1 = GATConv(in_channels=num_node_features, out_channels=self.hidden * 16, heads=self.heads, dropout=0.2, add_self_loops=False, concat=False) #self-loops should already be added
        #self.p_fc1 = nn.Linear(self.hidden * 16, output_dim)

        self.relu = nn.LeakyReLU() #hyperparameter, currently 0.01
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        #self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(256, self.n_output) #smaller now
    def forward(self, p1_data, p2_data):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch

        x = self.conv1(x=p1_x, edge_index = p1_edge_index) #need integrate edge feat
        
	    # global pooling
        x = gep(x, p1_batch)  
       
        # flatten
        #x = self.relu(self.p1_fc1(x))
        x = self.relu(x)
        x = self.dropout(x)

        xt = self.conv1(x = p2_x, edge_index = p2_edge_index)
	
	    # global pooling
        xt = gep(xt, p2_batch)  #bugging

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)
	
	    # Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        #xc = self.fc2(xc)
        #xc = self.relu(xc)
        #xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out #in [0,1]
class GAT_plm_edgefeat(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024, output_dim=128, dropout=0.2, heads = 1): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GAT plm edgefeat')
        super(GAT_plm_edgefeat, self).__init__()

        self.hidden = 8 #hyperparameter, though 8 seems to perform best
        self.heads = heads #multi head attention? try performance w/ more heads (3?)

        self.n_output = n_output
        self.conv1 = GATConv(in_channels=num_node_features, out_channels=self.hidden * 16, heads=self.heads, dropout=0.2, add_self_loops=False, concat=False) #self-loops should already be added
        #self.p_fc1 = nn.Linear(self.hidden * 16, output_dim)

        self.relu = nn.LeakyReLU() #hyperparameter, currently 0.01
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        #self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(256, self.n_output) #smaller now
    def forward(self, p1_data, p2_data):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch

        x = self.conv1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat) #need integrate edge feat
        
	    # global pooling
        x = gep(x, p1_batch)  
       
        # flatten
        #x = self.relu(self.p1_fc1(x))
        x = self.relu(x)
        x = self.dropout(x)

        xt = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
	
	    # global pooling
        xt = gep(xt, p2_batch)  #bugging

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)
	
	    # Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        #xc = self.fc2(xc)
        #xc = self.relu(xc)
        #xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out #in [0,1]
class GAT_plm_dssp(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, output_dim=128, dropout=0.2, heads = 1): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GAT plm dssp')
        super(GAT_plm_dssp, self).__init__()

        self.hidden = 8 #hyperparameter, though 8 seems to perform best
        self.heads = heads #multi head attention? try performance w/ more heads (3?)

        self.n_output = n_output
        self.conv1 = GATConv(in_channels=num_node_features, out_channels=self.hidden * 16, heads=self.heads, dropout=0.2, add_self_loops=False, concat=False) #self-loops should already be added
        #self.p_fc1 = nn.Linear(self.hidden * 16, output_dim)

        self.relu = nn.LeakyReLU() #hyperparameter, currently 0.01
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        #self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(256, self.n_output) #smaller now
    def forward(self, p1_data, p2_data):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch

        x = self.conv1(x=p1_x, edge_index = p1_edge_index) #need integrate edge feat
        
	    # global pooling
        x = gep(x, p1_batch)  
       
        # flatten
        #x = self.relu(self.p1_fc1(x))
        x = self.relu(x)
        x = self.dropout(x)

        xt = self.conv1(x = p2_x, edge_index = p2_edge_index)
	
	    # global pooling
        xt = gep(xt, p2_batch)  #bugging

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)
	
	    # Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        #xc = self.fc2(xc)
        #xc = self.relu(xc)
        #xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out #in [0,1]
class GAT_plm_dssp_edgefeat(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, output_dim=128, dropout=0.2, heads = 1): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GAT plm dssp edgefeat')
        super(GAT_plm_dssp_edgefeat, self).__init__()

        self.hidden = 8 #hyperparameter, though 8 seems to perform best
        self.heads = heads #multi head attention? try performance w/ more heads (3?)

        self.n_output = n_output
        self.conv1 = GATConv(in_channels=num_node_features, out_channels=self.hidden * 16, heads=self.heads, dropout=0.2, add_self_loops=False, concat=False) #self-loops should already be added
        #self.p_fc1 = nn.Linear(self.hidden * 16, output_dim)

        self.relu = nn.LeakyReLU() #hyperparameter, currently 0.01
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        #self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(256, self.n_output) #smaller now
    def forward(self, p1_data, p2_data):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch

        x = self.conv1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat) #need integrate edge feat
        
	    # global pooling
        x = gep(x, p1_batch)  
       
        # flatten
        #x = self.relu(self.p1_fc1(x))
        x = self.relu(x)
        x = self.dropout(x)

        xt = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
	
	    # global pooling
        xt = gep(xt, p2_batch)  #bugging

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)
	
	    # Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        #xc = self.fc2(xc)
        #xc = self.relu(xc)
        #xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out #in [0,1]
class GAT_plm_dssp_edgefeat_sagpool(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GAT-plm-dssp-edgefeat: sagpool edition')
        super(GAT_plm_dssp_edgefeat_sagpool, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATConv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #20%

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]

#Ablation II: Graph Convolution Mechanism
class GAT_sagpool_baseline(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GAT-sagpool-baseline')
        super(GAT_sagpool_baseline, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATConv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #20%

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class GCN_sagpool(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GCN-sagpool')
        super(GCN_sagpool, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GCNConv(in_channels=num_node_features, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #20%

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index)
        p1_x = self.conv2(x = p1_x, edge_index = p1_edge_index)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index)
        p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class GATv2_sagpool(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded GATv2: sagpool edition')
        super(GATv2_sagpool, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class MH_GATv2_sagpool(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH GATv2: sagpool edition')
        super(MH_GATv2_sagpool, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class MH_GT_sagpool(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH Graph Transformer: sagpool edition')
        super(MH_GT_sagpool, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = TransformerConv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class GT_sagpool(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded Graph Transformer: sagpool edition')
        super(GT_sagpool, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = TransformerConv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]

#Model for 5-Fold CV
class MH_GATv2_sagpool_GraphConv(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH GATv2: sagpool edition, GraphConv')
        super(MH_GATv2_sagpool_GraphConv, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.out = torch.nn.Linear(self.hidden_dim//2, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.1) #hyperparameter: set to 0.1 and test
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.lin2(xc))
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class MAPLEGNN(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MAPLE-GNN')
        super(MAPLEGNN, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.out = torch.nn.Linear(self.hidden_dim//2, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.1) #hyperparameter: set to 0.1 and test
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.lin2(xc))
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]

#Unused Models
class MH_GATv2_sagpool_GraphConv_exp(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH GATv2: sagpool edition, GraphConv [experimental]')
        super(MH_GATv2_sagpool_GraphConv_exp, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.out = torch.nn.Linear(self.hidden_dim//2, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.1) #hyperparameter: set to 0.1 and test
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.lin2(xc))
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        return out
class MH_GATv2_sagpool_GCNConv(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH GATv2: sagpool edition, GCNConv')
        super(MH_GATv2_sagpool_GCNConv, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout, GNN=GCNConv) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class MH_GATv2_sagpool_GATConv(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH GATv2: sagpool edition, GATConv')
        super(MH_GATv2_sagpool_GATConv, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout, GNN=GATConv) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class MH_GATv2_sagpool_GATv2Conv(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH GATv2: sagpool edition, GATv2Conv')
        super(MH_GATv2_sagpool_GATv2Conv, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout, GNN=GATv2Conv) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]
class MH_GATv2_sagpool_SAGEConv(nn.Module):
    def __init__(self, n_output=1, num_node_features = 1024+14, hidden_dim = 128, dropout=0.2, pool_dropout= 0.2): 
        #num_node_features: 1038 for PLM+DSSP, 2256 for PLM+1DMF+DSSP, reduced output dim for space constraint, now 256 for extra features, 3 head attention
        #do somoething here
        print('Loaded MH GATv2: sagpool edition, SAGEConv')
        super(MH_GATv2_sagpool_SAGEConv, self).__init__()

        self.hidden_dim = hidden_dim #hyperparameter, though 8 seems to perform best
        self.pool_dropout = pool_dropout
        #self.min_score = min_score
        # protein 1 & protein 2
        self.n_output = n_output

        self.conv1 = GATv2Conv(in_channels=num_node_features, out_channels=hidden_dim, dropout=dropout, edge_dim=2, concat=False, heads=3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.sagpool1 = SAGPooling(in_channels=hidden_dim, ratio=self.pool_dropout, GNN=SAGEConv) #30%

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=0.0, edge_dim=2)

        #self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=2)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        #self.sagpool2 = SAGPooling(in_channels=hidden_dim, ratio=self.po) #0.125 of nodes

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.lin2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.n_output)

        self.relu = nn.LeakyReLU(negative_slope=0.01) #hyperparameter
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, p1_data, p2_data, train=True):
        p1_x, p1_edge_index, p1_edge_feat, p1_batch = p1_data.x, p1_data.edge_index, p1_data.edge_attr, p1_data.batch
        p2_x, p2_edge_index, p2_edge_feat, p2_batch = p2_data.x, p2_data.edge_index, p2_data.edge_attr, p2_data.batch
        
        p1_x = self.conv1(x = p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat)
        p1_x = self.relu(p1_x)
        p1_x = self.bn1(p1_x)
        p1_x = self.dropout(p1_x)
        p1_x, p1_edge_index, p1_edge_feat, p1_batch, p1_perm, p1_attr = self.sagpool1(x=p1_x, edge_index = p1_edge_index, edge_attr = p1_edge_feat, batch=p1_batch) #x, edge_idx, edge_feat, batch, perm, scores
        #p1_x = self.dropout(p1_x)
        p1_x1 = gep(p1_x, p1_batch)
        #print(self.sagpool1(p1_x, p1_edge_index, p1_edge_feat, p1_batch))
        
        #p1_x1 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1) 
        #p1_x2 = torch.cat([gmp(p1_x, p1_batch), gep(p1_x, p1_batch)], dim=1)

        p2_x = self.conv1(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        p2_x = self.relu(p2_x)
        p2_x = self.bn1(p2_x)
        p2_x = self.dropout(p2_x)
        p2_x, p2_edge_index, p2_edge_feat, p2_batch, p2_perm, p2_attr = self.sagpool1(x=p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat, batch=p2_batch)
        #p2_x = self.dropout(p2_x)
        #p2_x1 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)
        p2_x1 = gep(p2_x, p2_batch)

        #p2_x = self.conv2(x = p2_x, edge_index = p2_edge_index, edge_attr = p2_edge_feat)
        #p2_x = self.relu(p2_x)
        #p2_x2 = torch.cat([gmp(p2_x, p2_batch), gep(p2_x, p2_batch)], dim=1)

        #p2_x = p2_x1 + p2_x2

        xc = torch.cat((p1_x1, p2_x1), 1)
        #print("done conv")
        #print("done mutual attn")
        # add some dense layers
        xc = self.relu(self.lin1(xc))
        xc = self.dropout(xc)
        #xc = self.relu(self.lin2(xc))
        #xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        if (train):
            return out
        else:
            return out, p1_perm, p1_attr, p2_perm, p2_attr #in [0,1]

model = MAPLEGNN()

print(model)