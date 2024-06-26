# clark-scholars-ppi-predict
research project for the **Clark Scholars Program** (currently private)\
building LLM-augmented ML model for PPI predictions incorporating both structure and sequence data from protein sequences\
**EXACT PROJECT DETAILS STILL IN PROGRESS**
learning machine learning by doing (have not coded a single line of ML prior to this project) \
\
*Relevant Information:* 
Old dataset (may not use)

Dataset is curated from Richoux et. al. (2019), https://arxiv.org/pdf/1901.06268, which features a curated PPI prediction database specifically aimed to prevent Information Leak, used for SOTA benchmarks in models like SPNet \
Dataset prevents overfitting in a three-fold manner. There are three separate datasets: Training, Validation, and Testing. Training is for training the model's parameters. Validation is for validating the model and tuning its hyperparameters. Testing is for final testing after both parameters and hyperparameters are set. **Importantly, no protein sequence in any of the datasets will show up in any other dataset.** This setup is more reliable than cross-validation due to this feature, as the model must predict protein-protein interaction without any prior knowledge on either of the proteins.
