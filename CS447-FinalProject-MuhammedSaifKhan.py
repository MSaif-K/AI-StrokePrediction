# -*- coding: utf-8 -*-
"""
Name: Muhammed Saif Khan
Student ID: 190201120
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import missingno as ms
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# Training dataset.
train = pd.read_csv('healthcare-dataset-stroke-data.csv').sample(frac = 1)

###########################
# ~ Data Preprocessing ~ #
###########################

print('''~~~~~~~~~~~~~~~~~~~~~~
~ Data Preprocessing ~
~~~~~~~~~~~~~~~~~~~~~~
''')

print('''Training dataset head:
=======================
{0}'''.format(train.head()))                    # Sample data from training dataset.

train_null_vals = train.isnull().sum()          # Null-value counts in training dataset.
print('''Null values in training dataset:
=================================
{0}\n\n'''.format(train_null_vals))

# Null value matrix plot for training and testing datasets before removing null values.
plt.figure(1)
ms.matrix(train)
plt.title('Null Values In Training Dataset', fontsize = 32)
plt.show()
plt.close(fig = plt.figure(1))

# Removing null values from both datasets.
train_c = train.dropna(axis = 0, how = 'any')

# Shape of training and testing datasets before and after removing null values.
print('\nTraining dataset shape before and after removing null values:\n{0} -> {1}'.format(
    train.shape, train_c.shape))

# Null value matrix plot for training and testing datasets after removing null values.
plt.figure(2)
ms.matrix(train_c)
plt.title('Null Values In Training Dataset', fontsize = 32)
plt.show()
plt.close(fig = plt.figure(2))

# Comparing number of patients with and without strokes and displaying as a matrix.
print('''\n\nValue count matrix for stroke affected patients:
=================================================
{0}\n'''.format(train_c['stroke'].value_counts()))

# Plotting above comparison as a bar graph.
plt.figure(3)
sb.countplot(x = train_c['stroke'])
plt.title('No of Patients Affected by Stroke', fontsize = 16)
plt.show()
plt.close(fig = plt.figure(3))

'''Comparing number of patients with and without strokes
grouped by gender and displaying as a matrix.'''
print('''Value count matrix for stroke affected patients grouped by gender:
===================================================================
{0}\n'''.format(train_c.groupby(['gender'])['stroke'].value_counts()))

# Plotting bar graph of comparison of gender vs stroke.
plt.figure(4)
sb.countplot(x = train_c['gender'], hue = train_c['stroke'])
plt.title('Gender vs Stroke', fontsize = 16)
plt.show()
plt.close(fig = plt.figure(4))

# Comparing number of patients' smoking status and displaying as a matrix.
print('''Value count matrix of patients\' smoking status:
================================================
{0}\n'''.format(train_c['smoking_status'].value_counts()))

# Plotting above comparison as a bar graph.
plt.figure(5)
sb.countplot(x = train_c['smoking_status'])
plt.title('Types of Smokers', fontsize = 16)
plt.show()
plt.close(fig = plt.figure(5))

'''Comparing number of patients\' smoking status
grouped by gender and displaying as a matrix.'''
print('''Value count matrix of patients\' smoking status grouped by gender:
==================================================================
{0}\n'''.format(train_c.groupby(['gender'])['smoking_status'].value_counts()))

# Plotting bar graph of comparison of gender vs types of smokers.
plt.figure(6)
sb.countplot(x = train_c['gender'], hue = train_c['smoking_status'])
plt.title('Gender vs Types of Smokers', fontsize = 16)
plt.show()
plt.close(fig = plt.figure(6))

'''Label encoding each column in the training and testing
datasets from str values to int values.'''
le = LabelEncoder()
train_p = train_c.copy()
for i in train_c.columns:
    train_p[i] = le.fit_transform(train_c[i])

# Min-Max scaling.
minimum = train_p.min(axis='rows', numeric_only=(True)).drop(['id'])
maximum = train_p.max(axis='rows', numeric_only=(True)).drop(['id'])

train_p = ((train_p - minimum) / (maximum - minimum)).dropna(axis=1)
# =============================================================================
# =============================================================================


######################
# ~ Model Creation ~ #
######################

print('''
~~~~~~~~~~~~~~~~~~
~ Model Creation ~
~~~~~~~~~~~~~~~~~~
''')

Xtrain = train_p.drop('stroke', axis = 1)
ytrain = train_p['stroke']
print('Xtrain shape: {0}\nytrain shape: {1}\n'.format(Xtrain.shape, ytrain.shape))

# Train-Test Split.
X_train, X_test, y_train, y_test = train_test_split(
    Xtrain, ytrain, test_size = 0.2, random_state = 42)

# Train-Test Split variable shapes.
print('X_train shape: {0}\nX_test shape: {1}\ny_train shape: {2}\ny_test shape: {3}'.format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# Model 1: Decision Tree
print('''
======================
Model 1: Decision Tree
======================''')

dtc = DecisionTreeClassifier(max_depth = 3)
dtc.fit(X_train, y_train)

dtc_y_pred = dtc.predict(X_test)

# Test score.
dtc_test_score = dtc.score(X_test, y_test)
print('Decision Tree test score: {0}\n'.format(dtc_test_score))

# Classification report.
dtc_report = classification_report(y_test, dtc_y_pred, zero_division = 1)
print('Decision Tree classification report:\n{0}'.format(dtc_report))

# Confusion matrix.
dtc_cm = confusion_matrix(y_test, dtc_y_pred)
print('Decision Tree confusion matrix:\n{0}\n'.format(dtc_cm))

# Cross Validation score.
dtc_cv = cross_validate(dtc, Xtrain, ytrain)
print('Mean 5-Fold Cross Validation score For Decision Tree: {0}'.format(
    statistics.mean(list(dtc_cv['test_score']))))
# =============================================================================


# Model 2: Random Forest
print('''
======================
Model 2: Random Forest
======================''')

rfc = RandomForestClassifier(n_estimators = 75)
rfc.fit(X_train, y_train)

rfc_y_pred = rfc.predict(X_test)

# Test score.
rfc_test_score = rfc.score(X_test, y_test)
print('Random Forest test score: {0}\n'.format(rfc_test_score))

# Classification report.
rfc_report = classification_report(y_test, rfc_y_pred, zero_division = 1)
print('Random Forest classification report:\n{0}'.format(rfc_report))

# Confusion matrix.
rfc_cm = confusion_matrix(y_test, rfc_y_pred)
print('Random Forest confusion matrix:\n{0}\n'.format(rfc_cm))

# Cross Validation score.
rfc_cv = cross_validate(rfc, Xtrain, ytrain)
print('Mean 5-Fold Cross Validation score for Random Forest: {0}'.format(
    statistics.mean(list(rfc_cv['test_score']))))
# =============================================================================
# =============================================================================


####################################
# ~ Principal Component Analysis ~ #
####################################

print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ Principal Component Analysis ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''')

# Applying PCA on both models.
pca = PCA(n_components = 7)
p_components = pca.fit_transform(Xtrain)

# Train-Test Split.
X_train, X_test, y_train, y_test = train_test_split(
    Xtrain, ytrain, test_size = 0.25, random_state = 42)

# Train-Test Split variable shapes.
print('X_train shape: {0}\nX_test shape: {1}\ny_train shape: {2}\ny_test shape: {3}'.format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# Decision Tree after PCA
# ========================
print('''
=============================
Applying PCA on Decision Tree
=============================''')
dtc = DecisionTreeClassifier(max_depth = 3)
dtc.fit(X_train, y_train)

dtc_y_pred = dtc.predict(X_test)

# Test score.
dtc_test_score = dtc.score(X_train, y_train)
print('Decision Tree test score after PCA: {0}\n'.format(dtc_test_score))

# Classification report.
dtc_report = classification_report(y_test, dtc_y_pred, zero_division = 1)
print('Decision Tree classification report after PCA:\n{0}'.format(dtc_report))

# Confusion matrix.
dtc_cm = confusion_matrix(y_test, dtc_y_pred)
print('Decision Tree confusion matrix after PCA:\n{0}\n'.format(dtc_cm))

# Cross Validation score.
dtc_cv = cross_validate(dtc, Xtrain, ytrain)
print('Mean 5-Fold Cross Validation score for Decision Tree after PCA: {0}'.format(
    statistics.mean(list(dtc_cv['test_score']))))

# Random Forest after PCA
# ========================
print('''
=============================
Applying PCA on Random Forest
=============================''')
rfc = RandomForestClassifier(n_estimators = 75)
rfc.fit(X_train, y_train)

rfc_y_pred = rfc.predict(X_test)

# Test score.
rfc_test_score = rfc.score(X_test, y_test)
print('Random Forest test score after PCA: {0}\n'.format(rfc_test_score))

# Classification report.
rfc_report = classification_report(y_test, rfc_y_pred, zero_division = 1)
print('Random Forest classification report after PCA:\n{0}'.format(rfc_report))

# Confusion matrix.
rfc_cm = confusion_matrix(y_test, rfc_y_pred)
print('Random Forest confusion matrix after PCA:\n{0}\n'.format(rfc_cm))

# Cross Validation score.
rfc_cv = cross_validate(rfc, Xtrain, ytrain)
print('Mean 5-Fold Cross Validation score for Random Forest after PCA: {0}'.format(
    statistics.mean(list(rfc_cv['test_score']))))
