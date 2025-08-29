# Predicting Abalone Age using Neural Networks

## Description
Project developed for the Artificial Intelligence course. The goal is to estimate the age of an abalone based on its physical characteristics using a Multi-Layer Perceptron (MLPRegressor) from scikit-learn.

## Dataset
- Abalone Dataset: 4177 samples, 8 features + target
- Features: Sex, Length, Diameter, Height, WholeWeight, ShuckedWeight, VisceraWeight, ShellWeight
- Target: Rings (number of shell rings, used to estimate age)

## Methodology
1. Split data: 75% train, 25% test  
2. Feature standardization  
3. Train MLPRegressor  
4. Hyperparameter optimization with GridSearchCV  
5. Evaluation with metrics and plots  

## Results
- Best model: hidden_layer_sizes=(200,), learning_rate=0.01  
- MSE: ~4.13  
- MAE: ~1.47 rings (~2.2 years)  
- R² ≈ 0.6  

## Author
Bostan Mihai-Tudor, 2025
