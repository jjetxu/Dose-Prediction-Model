# Overview

This project implements a preoperative anesthetic dosage prediction model using PyTorch. It is designed as a learning-focused exploration of machine learning and neural networks in a clinical context, using real-world structured patient data. The goal is to predict an appropriate preoperative anesthetic dose for colonoscopy patients based on basic demographic and physiological variables.

The project emphasizes correct ML workflow rather than clinical deployment, including data cleaning, normalization, train/test separation, nonlinear modeling, and evaluation.

Model Description

The current model is a feedforward artificial neural network (ANN) trained for regression. It takes a small set of routinely available preoperative features and outputs a continuous dose prediction.

Input features
 - Sex (binary encoded)
 - Age
 - Weight
 - Systolic blood pressure (SBP)
 - Diastolic blood pressure (DBP)

Target
 - Preoperative anesthetic dose (mg)

The network architecture is:

Input layer → 16 hidden units → 8 hidden units → output

SELU activation functions

Adam optimizer with L2 regularization

This nonlinear architecture allows the model to capture nonlinear relationships and feature interactions that cannot be represented by linear regression, which aligns with findings in the referenced literature.

Data Handling

Rows with missing required values are removed.

Patients with zero-dose records are excluded for regression stability.

Data is shuffled and split into training and test sets (80/20).

Numeric inputs are normalized using training-set statistics only, preventing data leakage.

Purpose

This project is not intended for clinical use, but as a foundation for understanding how ML models can support anesthetic decision-making. It provides a base for future extensions such as additional features, uncertainty estimation, or intraoperative time-series modeling.