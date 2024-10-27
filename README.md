# Diabetes-Prediction

 This project aimed to develop a predictive model for diabetes diagnosis by analyzing medical data, focusing on key health metrics associated with diabetes

 The project employed two machine learning algorithms—linear Regression and K-Nearest Neighbors (KNN)—to predict the likelihood of diabetes in patients. The models were trained and evaluated on a dataset 
 containing various patient metrics, including blood pressure, insulin levels, BMI, age, and family history of diabetes.

1. Linear Regression: Initially, we used Linear Regression as a baseline predictive model. However, its performance yielded a lower accuracy score of 0.33, indicating limited reliability in accurately predicting 
   diabetes.

2. K-Nearest Neighbors (KNN): We implemented a KNN model to enhance predictive accuracy. KNN achieved a significantly better result with an accuracy score of 0.779, indicating a much stronger correlation between 
   patient health metrics and diabetes diagnosis.

3. Confusion Matrix Analysis: A confusion matrix was generated for each model to further analyze prediction performance. The confusion matrix for KNN demonstrated a better balance between true positives and true 
   negatives, showcasing its effectiveness in distinguishing diabetic and non-diabetic cases more accurately than Linear Regression.

Results: The KNN model’s 77.9% accuracy demonstrated that it is a more suitable algorithm for predicting diabetes in this dataset, outperforming Linear Regression significantly. Future work could involve hyperparameter tuning and exploring additional models to improve prediction accuracy further.

# Requirements
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=27) # best option 27
from sklearn import metrics
