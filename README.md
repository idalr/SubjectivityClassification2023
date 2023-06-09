# SubjectivityClassification2023

This repository was developed as part of the CogSys Master's project module (PM1: Mining opinions and arguments) at the University of Potsdam, Germany.

## Reproduction Instructions

1. To reproduce the evaluation results, please run the evaluation.py file. This could take up to 10-20 mins to complete
2. To reproduce the model training, please refer to the notebook related to the models of interest inside the project folder

## Project description

The goal of this project is to reproduce and extend the results from [Alhindi et al. (2020)](https://aclanthology.org/2020.coling-main.540.pdf), by using sentence-level argumentation strategies as the key feature to classify news articles as subjective (correspond to editorial, op-ed, and letter) or objective (correspond to news report).

## Project corpora

This project uses data from two corpora:
1. [Webis-editorial-6](https://webis.de/data/webis-editorials-16.html)
2. [NYTAC](https://catalog.ldc.upenn.edu/LDC2008T19)

Files of extracted features NYTAC are provided in this repository to allow the reproduction of the models' evaluation, along with the notebook (See data/ProcesssNYT.ipynb and data/GetAllNYTFeat.ipynb) in which the processing steps took place.

## Models
Three groups of models are trained in the project:
1. end-to-end models (See project/End-to-endExperment/End-to-end experiment.ipynb)
2. single-feature models (See project/SingleFeatureRNNs-on-NYTAC.ipynb)
3. combined-feature models (See project/CombinedModels-on-NYTAC.ipynb and CombinedModels-on-Webis-16.ipynb)

Weights of the trained models in 2. and 3. are stored in project/ModelWeights for the reproduction purpose

## Results and analysis
The results of the feature-based RNN models are stored in the project/Results, and the linguistic analysis of the NYTAC data is stored in data/LinguisticAnalysis.ipynb with the graphs in data/Images

Last update: 14.04.2023
