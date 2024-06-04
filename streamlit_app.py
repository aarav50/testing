# S10.1: Copy this code cell in 'iris_app.py' using the Sublime text editor. You have already created this ML model in the previous class(es).

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)


def perdiction(sw,sl,pw,pl):
    answer=svc_model.predict([[sl,sw,pl,pw]])

    if answer[0]==0:
            return('Iris-setosa')
    elif answer[0]==1:
            return('Iris-virginica')
    elif answer[0]==2:
            return('Iris-versicolor')

sepall=st.slider('sepl l',0.0,10.0)
sepalw=st.slider('sepl w',0.0,10.0)
petall=st.slider('petal l',0.0,10.0)
petalw=st.slider('petal w',0.0,10.0)
a=st.button('click')
if a == True:
    st.write('flower is ', perdiction(sepall, sepalw, petall, petalw))
    st.write('the score is ', score)


