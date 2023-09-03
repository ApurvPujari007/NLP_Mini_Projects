#!/usr/bin/env python
# coding: utf-8

# In[2]:
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


# Data cleaning
def data_clean(data,columns):  
    stop_words=set(stopwords.words("english"))
    def clean_text(doc):
        text=doc.split()
        text=[re.sub('[\[\]\}\{\:\,\.\:\']', '', x) for x in text]
        text=[x.lower() for x in text]
        text=" ".join([x for x in text if x not in stop_words])
        return text

    for col in columns:
        for i in range(data.shape[0]):
            data[col][i]=clean_text(data[col][i])
    return data


# In[3]:


# Label Encoding 
def label_Encode(data,tgt_col):
    ll=LabelEncoder()
    data[tgt_col] = ll.fit_transform(data[tgt_col])
    return data,ll


# In[4]:


# Aggregating every collunm 
def aggregation(data,columns,tar_col):
    for i in range(data.shape[0]):
        for col in columns:
            data[tar_col][i]+=data[col][i]
    # Dropping reduntant collunms
    data.drop(columns,axis=1,inplace=True)
    return data


# In[5]:


# Converting to string to sequence
def to_seq(data,column):
    def make_dictionary(data,column):
        doc=" "
        for i in range(data.shape[0]):
            doc =doc+" "+data[column][i]
        # Selecting only alphabetic words for tokenization
        words=word_tokenize(str(doc))
        words=list(set([word for word in words if word.isalpha()]))
        # Creating 2 dictionaries {word,index} {index,word}
        word_num={}
        num_word={}
        for i in range(len(words)):
            word_num.setdefault(words[i],i)
            num_word.setdefault(i,words[i])
        return word_num,num_word
    
    word_num,num_word=make_dictionary(data,column)
    
    # Function to map that dictionary to every string
    def tonum(string):
        num=[]
        tokens=string.split(" ")
        tokens=[x for x in tokens if x.isalpha()]
        for i in tokens:
            for k,v in word_num.items():
                if k==i:
                    num.append(v)
        return num
    
    # Applyaing tonum() 
    data[column]=data[column].apply(lambda x:tonum(x))
    return data


# In[16]:


# to balance data
def cat_balance(data,col):
# data representation part -1   
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.countplot(x=col,data=data);
    plt.title("Before oversampling")
#Making dataframe for unique values   
    df=pd.DataFrame(data[col].value_counts())
    df["cata"]=df.index
    max_cat=df[col].max()
    df["balance_by"]=max_cat-df[col]
  
    group_cat=data.groupby(col)
    group_keys=list(group_cat.groups.keys())
    
    for i in range(df.shape[0]):
        numbers=group_cat.get_group(group_keys[i]).index
        num_values=df["balance_by"][i]
#Randomly select values from the list
        random_values = np.random.choice(numbers, size=num_values, replace=True)
        data = pd.concat([data, data.iloc[random_values]], axis=0)
# data representation part -2
    plt.subplot(1,2,2)
    sns.countplot(x=col,data=data)
    plt.title("After oversampling");
    return data

def pad_X(X):
    max1=0
    for i in X:
        if len(i)>max1:
            max1=len(i)   # selecting maximum padding length 

    def padd(li):
        adj=max1-len(li)
        if adj>0:
            li.extend(np.full((1,adj), -1)[0].tolist())
            return li
        elif adj<0:
            print("error adj< max1")
            return None
        elif adj==0 :
            return li

    X.apply(lambda x:padd(x));
    return X


# In[17]:

#Preparing data
def dnn_to_xmatrix(data):
    xx=[]    
    for i in range(data.shape[0]):
        xx.append(data[i])
    return np.array(xx)
def dnn_to_ymatrix(data):
    xx=[]    
    for i in range(data.shape[0]):
        xx.append([data[i]])
    return np.array(xx)

# In[26]:


def plot_gr(hist_df):
    #Loss plot
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.plot(hist_df.index, hist_df.loss,label = "loss")
    plt.plot(hist_df.index, hist_df.val_loss,label = "val loss")
    plt.title("Errors")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    x_coo=hist_df[hist_df["val_loss"]==hist_df["val_loss"].min()].index.values[0]
    y_coo=hist_df[hist_df["val_loss"]==hist_df["val_loss"].min()]["val_loss"].values[0]
    plt.text(x_coo,y_coo,"*")
    plt.text(x_coo,y_coo,"min va_loss")
    plt.legend()
    plt.grid()

    # Accuracy PLot
    plt.subplot(1,2,2)
    plt.plot(hist_df.index, hist_df.accuracy,label = "accuracy")
    plt.plot(hist_df.index, hist_df.val_accuracy,label = "val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    x_coo=hist_df[hist_df["val_accuracy"]==hist_df["val_accuracy"].max()].index.values[0]
    y_coo=hist_df[hist_df["val_accuracy"]==hist_df["val_accuracy"].max()]["val_accuracy"].values[0]
    plt.text(x_coo,y_coo,"*")
    plt.text(x_coo,y_coo,"max val_accuracy")
    plt.legend()
    plt.grid()


# In[ ]:




