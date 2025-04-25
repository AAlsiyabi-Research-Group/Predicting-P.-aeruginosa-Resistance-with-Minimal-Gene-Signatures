# Predicting-P.-aeruginosa-Resistance-with-Minimal-Gene-Signatures
This repository contains the full analysis and codebase for the study: "Minimal Gene Signatures Enable High-Accuracy Prediction of Antibiotic Resistance in Pseudomonas aeruginosa".
We developed a Genetic Algorithm (GA) combined with AutoML models to identify minimal sets of ~35–40 genes capable of accurately predicting antibiotic resistance across four antibiotics (meropenem, ciprofloxacin, tobramycin, and ceftazidime) using transcriptomic profiles.
Achieved test accuracies range from 96–99%, meeting clinical diagnostic standards.

The repository includes:

Feature selection code (genetic_algorithm.py)

Machine learning training and evaluation notebooks (AutoML_clf_run.ipynb)

AutoML setup instructions (Autosklearn_Installation_guide.ipynb)

Data references (expression matrix and phenotypic classifications)

Conda environment file (GA_env.yml) for Genetic algorithm runs.

**How to Run:**


Use GA_env.yml and Autosklearn_Installation_guide.ipynb to create environment and install packages for the runs. 
The genetic_algorithm.py can be executed to perform feature selection.

Train machine learning models by running AutoML_clf_run.ipynb.

Requirements
Python 3.9
