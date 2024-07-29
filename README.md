# MAPLE-GNN
**Multi-Head, Protein Language Model-Enhanced Graph Neural Network for Protein-Protein Interaction Prediction**
First-author research project for the **Anson L. Clark Scholars Program** at Texas Tech University. Publication in preparation.
![MAPLE-GNN Architecture](https://github.com/btang2/MAPLE-GNN/blob/main/images/gnn-model-diagram.png?raw=true)
### Dataset
MAPLE-GNN was trained on the Struct2Graph 1:1 and 1:10 datasets available at [Struct2Graph](https://github.com/baranwa2/Struct2Graph). All necessary data files can be downloaded at [Zenodo](https://zenodo.org/records/13123920). \
Associated with each protein PDB ID is a preprocessed hybrid-feature graph representation consisting of three .npy files:
1. 'ID-node_feat_reduced_dssp.npy' (node feature matrix)
2. 'ID-edge_list_9.npy' (edge list generated using an angstrom cutoff of 9.0) 
3. 'ID-edge_feat_9.npy' (edge features generated using an angstrom cutoff of 9.0)
### Model Training and Testing
All code necessary for training, testing, and k-fold cross-validation is available at 'codebase/maplegnn'. 
### PPI Prediction
New PPI predictions, along with integrated gradients-extracted attribution scores, can be conducted under 'codebase/explanation/sagpool-explain.py'. Code for preprocessing new PDB files can be found under the 'ID_to_explain_graph' method at 'codebase/data processing/modules.py'. Downloaded PDB files must be input into the 'codebase/data/explain-pdb' file, and hybrid-feature graph representations will be stored in the 'codebase/data/explain-npy' folder.
