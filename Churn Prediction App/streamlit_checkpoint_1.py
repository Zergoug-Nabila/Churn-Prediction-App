# -*- coding: utf-8 -*-
"""Streamlit_checkpoint_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wGCnb1wKhMjl5kRI9JRXnsX5au9h6Fsw
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('Expresso_churn_dataset.csv')


#colomns to delete
colonnes_a_supprimer = ['user_id', 'ARPU_SEGMENT', 'TOP_PACK','TENURE','ON_NET' , 'ORANGE' , 'TIGO' ,
                        'ZONE1' ,'ZONE2','MRG']

# delete
data = data.drop(columns=colonnes_a_supprimer)


data = data.drop_duplicates()


# Handling missing values
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)


for column in data.select_dtypes(include=[np.number]).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Encoding categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data
X = data.drop('CHURN', axis=1)
y = data['CHURN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
import pickle
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
