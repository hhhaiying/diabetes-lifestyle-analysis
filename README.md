# Lifestyle Factors and Diabetes Risk: A Bayesian Analysis

This repository contains the code, data, and report for a Bayesian analysis 
of lifestyle behaviors and high diabetes risk in a Mexican adult population, 
using data from the 2024 National Health and Nutrition Survey (ENSANUT 2024).

## Research Questions

1. Which lifestyle behaviors act as the primary drivers of high diabetes risk? 
2. What is the relative contribution and uncertainty associated with each of these risk factors?
3. Under counterfactual scenarios, what is the expected shift in population-level risk if targeted behavioral changes were implemented?


## Repository Structure
    data/               # Raw datasets from Kaggle and ENSANUT 2024
    doc/                # ENSANUT 2024 survey questionnaires
    notebook/           # Jupyter notebook with full analysis
    report/             # Final report (PDF)
    .gitignore
    README.md

## Data

- **Diabetes_Mexico.csv**: Primary dataset integrating anthropometric 
measurements, biochemical test results, and diabetes risk 
classification from ENSANUT 2024
- **actividad_fisica_ensanut2024.csv**: Physical activity sub-module 
from ENSANUT 2024

## Methods

- Causal structure specification (DAG)
- Frequentist logistic regression 
- Bayesian logistic regression
- Counterfactual simulation of population-level interventions
