# Predict_Student_Dropout_And_Acad
# import numpy as np
Numpy is general-purpose array-processing in python. It provides a high-performance multimensional array object and tools for working with these arrays. It is the fundamental package for scientific computing with Python. Besides its obvious scientific uses, Numpy can also be used as an efficient multi-dimensional container of generic data.
# import pandas as pd
pandas is a fast, powerful, flexiable and easy to use open data analysis and manipulation tools. It is built on the Python programmming language and is used for data manipulation and analysis. Pandas provides data structures for efficiently storing large datasets and tools for working with them. It is particularly useful for working with tabular data such as spreadsheets or SQL tables.
# import matplotlib.pyplot as plt 
Matplotlib is a comprehensive library for creating static, animated and interactive visualizations in Python. It is a plotting library for Python programming language and its numerical mathematics extension Numpy. It provides an object-oriented APl for embedding plots into application using general-purpose GUI toolkits like Tkinter.
# import seaborn as sns
Seaborn is a library for making statistical graphics in Python. Its build on top of matplotlib and integrates closely with pandas data structures. Seaborn help you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots.
# from google.colab import files 
The google.colab module provides a way to upload and download files to and from your Google Drive account. The files module is used to interact with the file system in Colab. You can use the files.upload() function to upload files from your local machine to the Colab environment and the files.download() function to download files from the Colab environment to your local machine.
# df.head()
The df.head() function is used to display the first few rows of a pandas DataFrame. By default, it displays the first 5 rows of the DataFrame. You can pass an integer argument to the function to display a different number of rows. For example, df.head(10) would display the first 10 rows of the DataFrame.
# df.info()
The df.info() function is used to display a summary of a pandas DataFrame. It provides information about the DataFrame such as the number of rows and columns, the data types of each column, and the amount of memory used by the DataFrame. It is a useful function for quickly getting an overview of a DataFrame.
# df.descride()
The df.describe() function is used to display a statistical summary of a pandas DataFrame. It provides information such as the count, mean, standard deviation, minimum and maximum values, and the quartiles of the data. By default, it only displays the summary for the numerical columns in the DataFrame. You can pass an argument to the function to include the summary for all columns in the DataFrame.
# EDA
Exploratory Data Analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. EDA is used for seeing what the data can tell us before the modeling task. It is not only important but also an essential step in any machine learning project. Python provides many libraries for EDA such as Pandas, NumPy, Matplotlib and Seaborn
![EDA](https://github.com/Akram23679/Predict_Student_Dropout_And_Acad/assets/111181292/02e2a9a3-ceae-4a51-80ee-63bf5cef4f66)
![camp](https://github.com/Akram23679/Predict_Student_Dropout_And_Acad/assets/111181292/7f01428d-d374-4a92-9f3f-68ff90488837)
# sns.histplot(x='Unemployment rate',data=df,hue='Target')
![unemp](https://github.com/Akram23679/Predict_Student_Dropout_And_Acad/assets/111181292/da71c269-b0bf-4bcc-bd88-92838938258d)
# sns.histplot(x='Inflation rate',data=df,hue='Target')
![infra](https://github.com/Akram23679/Predict_Student_Dropout_And_Acad/assets/111181292/8ce7f7de-6b01-425c-895c-2b838a101ba2)
# from sklearn.model_selection import train_test_split
The train_test_split() function is used to split a dataset into training and testing sets for machine learning. It is part of the scikit-learn library and is used to evaluate the performance of machine learning models. The function takes as input the dataset and a test size, and returns four arrays: the training data, the testing data, the training labels, and the testing labels.
# Logistic Regression
Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). It is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.
# from sklearn.linear_model import LogisticRegression
The LogisticRegression() function is used to create a logistic regression model for machine learning. It is part of the scikit-learn library and is used to predict the probability of a binary outcome based on one or more predictor variables.
# from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
These functions are used to evaluate the performance of a machine learning model.
![image](https://github.com/Akram23679/Predict_Student_Dropout_And_Acad/assets/111181292/a6d49538-7f32-4531-a5da-2dfbd3c9e6cf)
Accuracy 0.7740112994350282
**Logistic Regression performs very well on this case Study**
# K-Nearst Neighbor
K-Nearest Neighbor is a machine learning algorithm used for classification and regression. It is a non-parametric method used for classification and regression.
# from sklearn.neighbors import KNeighborsClassifier
The KNeighborsClassifier() function is used to create a K-Nearest Neighbor model for machine learning.
![image](https://github.com/Akram23679/Predict_Student_Dropout_And_Acad/assets/111181292/c331127e-1704-4cea-9b4b-229ae5119a00)
Accuracy 0.6598870056497175
![k value](https://github.com/Akram23679/Predict_Student_Dropout_And_Acad/assets/111181292/7302d19e-32e4-4b57-a988-cbf8647f1231)
**K-Nearst Neighbor does not perform well on this case Study**
