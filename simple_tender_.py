# -*- coding: utf-8 -*-
"""Simple_tender_.ipynb
Importing libraries
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import warnings
warnings.filterwarnings("ignore")

from tender_functions import *

import nltk
nltk.download('stopwords')
nltk.download('punkt')

"""Data Loading"""

data=pd.read_csv("tender.csv")

data.head()

data.tender_category.value_counts()

"""
1.   These are unwanted columns -> Unnamed: 0.1","Unnamed: 0","tender_id"
2.   Target column is "tender_category"
3.   These categories are unbalanced

Thus removing unwanted columns and balancing the categories is required to remove bias

Preparing data for cleaning
"""

data.drop(["Unnamed: 0.1","Unnamed: 0","tender_id"],inplace=True,axis=1)
data1=data[data["tender_category"].notnull()]
data1.reset_index(drop =True,inplace=True)
data1.drop("boq_len",axis=1,inplace=True)

"""Preparing data for DNN model"""

# data_clean fucntion removes punctuations,stopwords and any non-alphanumeric characters from given columns
clean_data=data_clean(data1,["tender_description","tender_details","tender_category","boq_items","boq_details"])

# label_Encode function simply label encodes the given column
encoded_data,label_obj=label_Encode(clean_data,"tender_category")

#aggregation function combines every column into single column
columns=["tender_details","boq_items","boq_details"]
tar_col="tender_description"
agg_data=aggregation(encoded_data,columns,tar_col)

#to_seq funciton converts text into numerically represented sequences
seq_data=to_seq(agg_data,"tender_description")

#cat_balance this function balances target column by oversampling method
balance_data=cat_balance(seq_data,"tender_category")

"""Separating features and target

"""

X=balance_data.tender_description
y=balance_data.tender_category

#pda_X this function pads the sequences according to maximum length of sequence in the data
X=pad_X(X)

# Splitting data
xtrain,xtest,ytrain,ytest=train_test_split(X,y,shuffle=True,stratify=y,test_size=0.2)
# Resetting index
xtrain.reset_index(inplace=True,drop=True)
xtest.reset_index(inplace=True,drop=True)
ytrain.reset_index(inplace=True,drop=True)
ytest.reset_index(inplace=True,drop=True)

# converting each row to numpy array

for i in range(xtrain.shape[0]):
    xtrain[i]=np.array(xtrain[i])
for i in range(xtest.shape[0]):
    xtest[i]=np.array(xtest[i])
for i in range(ytrain.shape[0]):
    ytrain[i]=np.array(ytrain[i])
for i in range(ytest.shape[0]):
    ytest[i]=np.array(ytest[i])

#dnn_to_xmatrix and dnn_to_ymatrix converts intput numpy array into single matrix
model_xtrain_1=dnn_to_xmatrix(xtrain)
model_xtest_1=dnn_to_xmatrix(xtest)
model_ytrain_1=dnn_to_ymatrix(ytrain)
model_ytest_1=dnn_to_ymatrix(ytest)

"""### DNN model"""

dim=model_xtrain_1.shape[1]
# making model
model = Sequential()
model.add(Dense(3000, input_shape=(dim,), activation='tanh'))
model.add(Dense(5, activation = 'softmax'))

#compiling model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
#Fitting model
history=model.fit(model_xtrain_1,model_ytrain_1,epochs=100,batch_size=10, validation_data=(model_xtest_1,model_ytest_1),verbose=1)

#This function plots the loss and accuracy curves
res_df = pd.DataFrame(history.history)
plot_gr(res_df)

"""Model Evaluation"""

eva=model.evaluate(model_xtest_1,model_ytest_1,verbose=0);
print(f"loss :{round(eva[0]*100,2)}%  accuracy :{round(eva[1]*100,2)}% ")

"""Prediction"""

true_labels =label_obj.inverse_transform(model_ytest_1)
print(f"Predicted labels:           True lables\n")
for i in range(model_xtest_1.shape[0]):
    pred=model.predict(model_xtest_1[i].reshape(1,6307),verbose=0)
    y_pred=label_obj.inverse_transform([np.argmax(pred)])
    print(f"{y_pred}     ---->    {true_labels[i]}")

#-------------------------------------------------------------"""Now with Unbalanced Data"""----------------------------------------------------------------------#

data=pd.read_csv("tender.csv")

data.drop(["Unnamed: 0.1","Unnamed: 0","tender_id"],inplace=True,axis=1)
data1=data[data["tender_category"].notnull()]
data1.reset_index(drop =True,inplace=True)
data1.drop("boq_len",axis=1,inplace=True)

# data_clean fucntion removes punctuations,stopwords and any non-alphanumeric characters from given columns
clean_data=data_clean(data1,["tender_description","tender_details","tender_category","boq_items","boq_details"])

# label_Encode function simply label encodes the given column
encoded_data,label_obj=label_Encode(clean_data,"tender_category")

#aggregation function combines every column into single column
columns=["tender_details","boq_items","boq_details"]
tar_col="tender_description"
agg_data=aggregation(encoded_data,columns,tar_col)

#to_seq funciton converts text into numerically represented sequences
seq_data=to_seq(agg_data,"tender_description")

X=seq_data.tender_description
y=seq_data.tender_category

#pda_X this function pads the sequences according to maximum length of sequence in the data
X=pad_X(X)

# Splitting data
xtrain,xtest,ytrain,ytest=train_test_split(X,y,shuffle=True,stratify=y,test_size=0.2)
# Resetting index
xtrain.reset_index(inplace=True,drop=True)
xtest.reset_index(inplace=True,drop=True)
ytrain.reset_index(inplace=True,drop=True)
ytest.reset_index(inplace=True,drop=True)

# converting each row to numpy array

for i in range(xtrain.shape[0]):
    xtrain[i]=np.array(xtrain[i])
for i in range(xtest.shape[0]):
    xtest[i]=np.array(xtest[i])
for i in range(ytrain.shape[0]):
    ytrain[i]=np.array(ytrain[i])
for i in range(ytest.shape[0]):
    ytest[i]=np.array(ytest[i])

#dnn_to_xmatrix and dnn_to_ymatrix converts intput numpy array into single matrix
model_xtrain_1=dnn_to_xmatrix(xtrain)
model_xtest_1=dnn_to_xmatrix(xtest)
model_ytrain_1=dnn_to_ymatrix(ytrain)
model_ytest_1=dnn_to_ymatrix(ytest)

dim=model_xtrain_1.shape[1]
# making model
model = Sequential()
model.add(Dense(3000, input_shape=(dim,), activation='tanh'))
model.add(Dense(5, activation = 'softmax'))

#compiling model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
#Fitting model
history_unbalanced=model.fit(model_xtrain_1,model_ytrain_1,epochs=100,batch_size=10, validation_data=(model_xtest_1,model_ytest_1),verbose=0)

#This function plots the loss and accuracy curves
res_df = pd.DataFrame(history_unbalanced.history)
plot_gr(res_df)
