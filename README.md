# MLOps with MLflow

Welcome to the MLflow project! This repository is dedicated to demonstrating the use of MLflow for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
  - [Experiment Tracking](#experiment-tracking)
  - [Model Versioning](#model-versioning)
  - [Model Deployment](#model-deployment)
- [Requirements](#requirements)


## Introduction

This project provides an end-to-end example of how to manage machine learning workflows using MLflow. MLflow is an open-source platform for managing the complete machine learning lifecycle, from experimentation to deployment. It enables data scientists and engineers to track experiments, package code into reproducible runs, and deploy machine learning models to production.

## Features

### Experiment Tracking

MLflow's experiment tracking allows you to record and query experiments, including code, data, config, and results. In this project, we track:
- Hyperparameters used for training
- Metrics such as accuracy, precision, recall, etc.
- Artifacts like trained models, visualizations, and more

### Model Versioning

MLflow provides model versioning to manage and store different versions of models. Each model is logged with metadata, and you can easily compare different versions to select the best-performing one.

### Model Deployment

After the best model is identified, MLflow simplifies the process of deploying it. This project demonstrates:
- How to save and load models using MLflow
- Deploying models to a local environment or a remote server using MLflow’s model serving capabilities

## Requirements

To run this project, ensure you have the following packages installed:

- Python 3.x
- MLflow
- Scikit-learn
- Pandas
- Numpy
- Matplotlib

## Explanation of the Structure:
### data/:
Contains raw and processed datasets.  The raw/ folder is for original datasets, and processed/ contains cleaned and preprocessed data ready for modeling.
### scripts/: 
Python scripts for automating tasks like training, evaluation, and deployment. These scripts can be run independently or via MLflow.

###  models/: 
Stores the models generated during the project. Each model can have multiple versions, and this directory keeps them organized.

### mlruns/:
This folder is automatically created by MLflow to store all the metadata related to your experiments, such as parameters, metrics, and artifacts.

### requirements.txt:
Lists all the Python dependencies needed to run the project, which can be installed using pip install -r requirements.txt.



