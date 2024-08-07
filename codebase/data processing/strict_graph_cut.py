import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_set = set()

for data in data_list:
    pdb_set.add(data.strip().split('\t')[1].lower())

pdb_list = list(pdb_set)

id_to_idx = {}
for i in range(len(pdb_list)):
    id_to_idx[pdb_list[i]] = i
    #print(f"{pdb_list[i]} : {i}, ")
print("generating graph from PPI data...")
#generate graph
G = nx.Graph()
ppi_posneg_dict = {}
#adj_matrix = np.zeros((len(pdb_list), len(pdb_list)))
with open('interactions_data.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        prot_a = parts[0].lower()
        prot_b = parts[1].lower()

        ppi_posneg_dict[(prot_a, prot_b)] = int(parts[2]) 
        ppi_posneg_dict[(prot_b, prot_a)] = int(parts[2]) 

        prot_a_idx = id_to_idx[prot_a]
        prot_b_idx = id_to_idx[prot_b]
        if (parts[2] == "0"):
            G.add_edge(prot_a_idx, prot_b_idx, capacity=0.0)
        else:
            G.add_edge(prot_a_idx, prot_b_idx, capacity=1.0)
        #G.add_edge(prot_b_idx, prot_a_idx, capacity=1.0)
        #adj_matrix[prot_a_idx, prot_b_idx] = 1
        #adj_matrix[prot_b_idx, prot_a_idx] = 1
#pickle.dump(G, open('ppi_graph.pickle', 'wb'))
#adj_matrix = process_ppi_graph() #only do once

print("loading graph...")
#G = nx.from_numpy_array(adj_matrix)

#G = pickle.load(open('ppi_graph.pickle', 'rb'))
n_nodes = G.number_of_nodes()
for i in range(n_nodes):
    if i not in G.nodes():
        print(f"{i} not in nodes: {pdb_list[i]}")
        G.add_edge(i, i, capacity=1.0) #temp fix for graph generation
        

#try mincut; as expected, did not actually work
#print("generating mincut...")
#cut_value, partitions = nx.minimum_cut(G, 0, len(pdb_list) - 1) #can probably optimize s and t
#print(f"cut value: {cut_value}, partitions: {partitions}")

#try bisection method:
print("generating graph bisection 1...")
partition_1, partition_2 = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=5, weight='capacity')
print(f"total proteins: {G.number_of_nodes()}")
print(f"partition 1 len: {len(partition_1)}, partition 2 len: {len(partition_2)}")

print("generating graph bisection 2...")
G_sub = G.subgraph(partition_2)
partition_2_1, partition_2_2 = nx.algorithms.community.kernighan_lin_bisection(G_sub, max_iter=5, weight='capacity')
print(f"partition 2 len: {len(partition_2)}, partition 2_1 len: {len(partition_2_1)}, partition 2_2 len: {len(partition_2_2)}")

print("generating graph bisection 3...")
G_sub_sub = G.subgraph(partition_2_2)
partition_2_2_1, partition_2_2_2 = nx.algorithms.community.kernighan_lin_bisection(G_sub_sub, max_iter=5, weight='capacity')
#print(f"{partition_1} / {partition_2_1} / {partition_2_2}")

#remove edges, use partition_1 & partition_2_1 for train, partition_2_2 for test
print("removing edges...")
print(f"total edges: {len(G.edges())}")
edges_to_remove = []
for u, v, d in G.edges(data=True):
    u_train = True
    v_train = True 
    if (u in partition_2_2_2):
        u_train = False
    if (v in partition_2_2_2):
        v_train = False
    
    if (u_train != v_train):
        edges_to_remove.append((u,v))

for u,v in edges_to_remove:
    G.remove_edge(u,v)

print(f"edges remaining: {len(G.edges())}")

#verify no information leak
for u, v, d in G.edges(data=True):

    if ((u in partition_1 or u in partition_2_1 or u in partition_2_2_1) and v in partition_2_2_2):
        print(f"information leak: edge ({u},{v})")
    if ((v in partition_1 or v in partition_2_1 or v in partition_2_2_1) and u in partition_2_2_2):
        print(f"information leak: edge ({u},{v})")

#generate new strict training and testing sets
train_file = "train_interactions_data8.txt"
test_file = "test_interactions_data8.txt"
if os.path.exists(test_file):
    os.remove(test_file)
if os.path.exists(train_file):
    os.remove(train_file)

testset = open(test_file, "w")
trainset = open(train_file, "w")
test_pos = 0
test_neg = 0
train_pos = 0
train_neg = 0
for u, v, d in G.edges(data=True):
    try:
        p1 = pdb_list[u]
        p2 = pdb_list[v]
        interact = ppi_posneg_dict[(p1,p2)]
        if (u in partition_2_2):
            testset.write(p1.upper() + "\t" + p2.upper() + "\t" + str(interact) + "\n")
            if (interact == 0):
                test_neg += 1
            else:
                test_pos += 1
        else:
            trainset.write(p1.upper() + "\t" + p2.upper() + "\t" + str(interact) + "\n")
            if (interact == 0):
                train_neg += 1
            else:
                train_pos += 1
    except Exception as e:
        print(e)
testset.close()
trainset.close()
print(f"file: {train_file} / {test_file}")
print(f"trainset: {train_pos + train_neg} interactions ({train_pos} pos, {train_neg} neg)")
print(f"testset: {test_pos + test_neg} interactions ({test_pos} pos, {test_neg} neg)")
print(f"total interactions: {train_pos + train_neg + test_pos + test_neg}")
"""
#draw graph
print("drawing graph...")
pos = nx.spring_layout(G, iterations=25, threshold=0.001)
plt.figure(figsize=(36,36))

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
color_map = []

for i in range(G.number_of_nodes()):
    part_1 = False
    part_2 = False
    if i in partition_1:
        part_1 = True
        color_map.append(colors[0])
    if i in partition_2_1:
        part_2 = True
        color_map.append(colors[0])
    if i in partition_2_2:
        part_2 = True
        color_map.append(colors[1])
    if part_1 and part_2:
        print("node in 2 partitions: " + str(i))
    if not part_1 and not part_2:
        print("node in 0 partitions: " + str(i))

width = 0.05
node_size = 50
font_size = 4
nx.draw_networkx_edges(G=G, pos=pos,width=width)
nx.draw_networkx_nodes(G=G, pos=pos, node_color = color_map, node_size=node_size)
nx.draw_networkx_labels(G=G, pos=pos, font_size = font_size)
plt.suptitle("Partition of Struct2Graph PPI graph using Kernighan-Lin Bisection")
plt.box(False) #remove margin box
plt.show()
"""