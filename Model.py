# Author @ Pandey Hitesh


#!/usr/bin/env python
# coding: utf-8

# # This Project is regarding the Loan prediction using Machine Learning with KNN algorithms.

# In[1]:


# importing essential libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Data Visualization libraries
import seaborn as sns  # Data Visualization libraries
import warnings

warnings.filterwarnings('ignore')

# In[2]:


# reading csv file
loan_data = pd.read_csv(r"Training Dataset.csv")

# In[3]:


# printing first five rows of dataset
loan_data.head(5)

# In[4]:


loan_data.shape

# In[5]:


loan_data.describe()

# In[6]:


# Statistical summary of dataset
loan_data.info()

# In[7]:


# Check null values
# After analysing null values we can counter it.
# Also we can predict the neccesary changes to data soit can gives more accurate results.
loan_data.isnull().sum()

# In[8]:


loan_data.head(5)

# In[9]:


# Dropping the Loan ID as it is unnecessary

loan_data = loan_data.drop(columns=['Loan_ID'])

# Categorizing the data

# In[10]:


Categorized_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term',
                       'Credit_History', 'Property_Area']
Numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

# Printing the data to analzysed it.

print(f'Categorized_columns data: {Categorized_columns}')
print(f'Numerical_columns data: {Numerical_columns}')

# Analyzing the columns via Data Visualizations

# In[11]:


Figures, Axes = plt.subplots(4, 2, figsize=(12, 15))
for idx, cat_col in enumerate(Categorized_columns):
    row, col = idx // 2, idx % 2
    sns.countplot(x=cat_col, data=loan_data, hue='Loan_Status', ax=Axes[row, col])
plt.subplots_adjust(hspace=1)

# Analyzation from above chart and conclusions are as follows.
# Loan Approval Status: Approximately two-thirds of applicants have been approved for a loan.
# Sex: There are roughly three times as many men as women.
# Marital Status: Married applicants are more likely to be granted loans than unmarried applicants.
# Dependents: The majority of the population has no dependents and is therefore more likely to be approved for a loan.
# Education: About 5/6th of the population is a graduate, and graduates have a higher loan approval rate.
# Employment:Approximately 83.33% of the population is not self-employed.
# Loan Amount Term: The vast majority of loans are for 360 months (30 years).
# Credit History: Applicant with a credit history has a much better chance of being accepted.
# Property Type: Semi-urban applicants are more likely to be approved for loans.

# In[12]:


Figures, Axes = plt.subplots(1, 3, figsize=(12, 15))
for idx, num_col in enumerate(Numerical_columns):
    sns.boxplot(y=num_col, data=loan_data, x='Loan_Status', ax=Axes[idx])

print(loan_data[Numerical_columns].describe())
plt.subplots_adjust(hspace=1)

# In[13]:


# There is no significant relationship between Numeric Columns and Loan Approval Status.


# # Preprocessing Data:
# Preprocessing data is a crucial step in the machine learning pipeline that involves cleaning and transforming,raw data into a format suitable for training a machine learning model.
# The quality of the input data has a significant impact on the performance of the model,and preprocessing helps in addressing various issues and improving the overall data quality.
# For here we are using,
# Encoding Categorical Features and Imputing missing values

# Input data needs to be pre-processed before we feed it to model.
# Convert categorical variables into a numerical format that can be easily fed into machine learning models.
# This may involve one-hot encoding, label encoding, or other techniques depending on the nature of the data andthe requirements of the model.

# In[14]:


# Encoding Categorical Features.
# converts categorical variable(s) into dummy/indicator variables,
# which are binary (0 or 1) columns representing the presence or absence of a particular category.
# The loan_data has been change as it has envolve through out the process and categorized into two columns.
loan_data_encoded = pd.get_dummies(loan_data, drop_first=True)
loan_data_encoded.head(10)

# loan_data is a DataFrame with a column named 'loan_amount', CoapplicantIncome
# Exclude NaN values and set to 0

loan_data_encoded['LoanAmount'] = loan_data['LoanAmount'].fillna(0)
loan_data_encoded['LoanAmount'].fillna(0, inplace=True)
# print(loan_data)

# To convert all possible data to 0 from nan Values

loan_data_encoded['CoapplicantIncome'] = loan_data_encoded['CoapplicantIncome'].fillna(0)
loan_data_encoded['CoapplicantIncome'].fillna(0, inplace=True)

# Find mean values excluding 0 and set it to values(0) in the Loan Amount columns

Average_of_LoanAmount = loan_data_encoded['LoanAmount'].mean()
print(f' The average of Loan Amount is {Average_of_LoanAmount}')

Average_of_CoapplicantIncome = loan_data_encoded['CoapplicantIncome'].mean()
print(f' The average of Loan Amount is {Average_of_CoapplicantIncome}')

# Replace the 0's with the Average mean
loan_data_encoded['LoanAmount'] = loan_data_encoded['LoanAmount'].replace(0, Average_of_LoanAmount)
loan_data_encoded['CoapplicantIncome'] = loan_data_encoded['CoapplicantIncome'].replace(0, Average_of_CoapplicantIncome)
loan_data_encoded.head()


# In[15]:


def filtering_data(df, column_name):
    # Fill NaN values with 0
    df[column_name] = df[column_name].fillna(0)

    # Calculate the mean excluding 0 values
    average_value = df[df[column_name] != 0][column_name].mean()

    # Replace 0's with the calculated mean
    df[column_name] = df[column_name].replace(0, average_value)

    print(f'The average of {column_name} is {average_value}')


# Assuming loan_data_encoded is your DataFrame
# Define columns to process
columns_to_process = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Apply the function to each column using a for loop
for column in columns_to_process:
    filtering_data(loan_data_encoded, column)

# Display the updated DataFrame
loan_data_encoded.head(50)

# Main Programme

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# for testing. basically a testing tool
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# In[17]:


# Spliting the dataset into 2 parts training and testing
X = loan_data_encoded.iloc[:, 0:12]
y = loan_data_encoded.iloc[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# In[18]:


# Scaling the data or Feature Scaling

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# In[19]:


classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')

# In[20]:


classifier.fit(X_train, y_train)

# In[21]:


y_pred = classifier.predict(X_test)
y_pred

# In[22]:


Confusion_Matrix = confusion_matrix(y_test, y_pred)
print(Confusion_Matrix)

# In[23]:


print(accuracy_score(y_test, y_pred))

# In[24]:


print(f1_score(y_test, y_pred))

# In[25]:


print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Test F1 Score: ", f1_score(y_test, y_pred))
print("Confusion Matrix on Test Data")
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# In[ ]:




