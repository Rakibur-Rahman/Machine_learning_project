# Machine_learning_project
Hand_Note :
Machine Learning defines: It is a type of Artificial Intelligence that allows
software applications to learn from the data and become more accurate in
predicting outcomes without human intervenvention.
Types of Machine Learning:
1. Supervised
This is a process of an algorithm learning from the training dataset or labelled
data set.
2. Unsupervised
This is a process where a model is trained using an information which is not
labelled.
3. Reinforcement
Reinforcement learning is learning by interacting with a space or an
environment.
Scitkit-learn:
_Open-source library built on NumPy, SciPy & Matplotlib.
Install:
Commands: pip install scikit-learn or conda install scikit-learn. But if we use
Anaconda then Scikit-learn pre-installed on it.
Using Example:
From sklearn.family import Model
Example: From sklearn.linear_model import LinearRegression
Regression: Regression is the prediction of a numeric value and often takes
input as a continuous value.
Classification: Classification is the problem identifying to which set of
categories a new observation belongs. There are two types of Classification .
1. Binary Classification and 2. Multi Classification.
Some of Algorithms for Classification:
1. Decision Tree
2. Random Forest
3. Naiive Bayes Classifier
4. Support vector Machine: SVM is supervised machine learning algorithm
which can be used for both classification or regression challenges
Coding Steps:
1. set working directory
Command:
import os
.oschdir(‘path’)
2. import pakages
Import numpy as np
3. load dataset using pandas:
Pandas.read_csv(‘file name’)
4. Make X & Y and Encode Y by
From sklearn.preprocessing import LabelEncoder
LabelEncoder.fit_transform(Y)
5.divide dataset into train and test
6.fit model with train data and predict with test data
7.find confusion matrix for percetage
