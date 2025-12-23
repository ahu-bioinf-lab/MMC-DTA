# MMC-DTA

A **multi-modal contrastive learning-based drug–target affinity prediction model** for anti-parasitic drug discovery.

---

## Introduction

This repository provides the implementation of **MMC-DTA**, a multi-modal contrastive learning framework for drug–target affinity prediction.

## Dependencies

- [ESM-2](https://github.com/facebookresearch/esm/tree/main?tab=readme-ov-file)
- - **Notice:** You need download the pre-trained model ESM-2 to obtain the protein residue graphs and the initial embedding of protein sequences.
- [ChemBERTa-2](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM/tree/main)
- -**Notice:** You need download the pre-trained model ChemBERTa-2 to obtain the initial features of drug sequences.

## Installing

- python = 3.8.13
  - torch==2.1.0
  - torch-geometric==2.6.1
  - pip >= 20.0, < 23.3
  - numpy==1.23.2
  - networkx==3.1
  - pandas==2.0.3
  - pip==24.2
  - rdkit==2024.3.5
  - scikit-learn==1.3.2
    
## Data Description

- data/Parasite/fold_1: randomly partitioned FDA_IC50 dataset
- data/Parasite/fold_OOD1: parasitic data in FDA_IC50 dataset as independent test set
- data/Parasite/duplicate_sequence.csv: protein sequences in the FDA_IC50 dataset
- data/Parasite/duplicate_smiles.csv: drug sequences in the FDA_IC50 dataset
- data/Parasite/duplicate_smiles_embedding.csv: the initial embedding of drug sequences in the FDA_IC50 dataset
- data/Parasite/duplicate_sequence_embedding.csv: the initial embedding of protein sequences in the FDA_IC50 dataset
- pre_process/sequence_representations_parasite: protein residue maps in FDA_IC50 dataset
- -Note that due to limited memory, some pre-training data has not been uploaded. You can follow the above prompts to download the pre-training model and obtain the preparation data.


