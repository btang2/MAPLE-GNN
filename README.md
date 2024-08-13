# MAPLE-GNN
A **M**ulti-head **A**ttention, **P**rotein **L**anguage Model-**E**nhanced **G**raph **N**eural **N**etwork for Protein-Protein Interaction Prediction: \
First-author research project for the **[Anson L. Clark Scholars Program](https://www.depts.ttu.edu/honors/academicsandenrichment/affiliatedandhighschool/clarks/)** at Texas Tech University. Publication in preparation. \
![MAPLE-GNN Architecture](https://github.com/btang2/MAPLE-GNN/blob/main/images/gnn-model-diagram.png?raw=true)
### Dataset
Raw data was taken from the Struct2Graph dataset available at [Struct2Graph](https://github.com/baranwa2/Struct2Graph). All necessary files can be downloaded at [Zenodo](https://zenodo.org/records/13123920). \
MAPLE-GNN was tested on 10 strict train-test splits without information leak generated from the heuristic Kernighan-Lin bisection algorithm on the derived PPI graph. Associated with each protein PDB ID is a preprocessed hybrid-feature protein graph representation:
1. `ID-node_feat_reduced_dssp.npy` (node feature matrix)
2. `ID-edge_list_9.npy` (edge list generated using an angstrom cutoff of 9.0) 
3. `ID-edge_feat_9.npy` (edge features generated using an angstrom cutoff of 9.0)
### Model Training and Testing
All code necessary for training and testing is available at `codebase/maplegnn`. \
Preprocessed graph representations downloaded from Zenodo should be placed in the `codebase/data/npy` directory. \
Each strict train-test split can be accessed at `train_interactions_data{i}.txt` and `test_interactions_data{i}.txt`, respectively.
### PPI Prediction
`codebase/explanation/sagpool-explain.py` can be used for new PPI predictions and integrated gradients-extracted attribution scores. \
Code for preprocessing new PDB files can be found under the `ID_to_explain_graph` method at `codebase/data processing/modules.py`. \
Downloaded PDB files must be input into the `codebase/data/explain-pdb` file, and hybrid-feature graph representations will be stored in the `codebase/data/explain-npy` folder.
