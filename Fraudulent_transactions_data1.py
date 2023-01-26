#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # DATA PRE PROCESSING

# In[2]:


#Reading the data 
df=pd.read_csv('Fraud.csv')

#Check shape of data
df.shape


# In[3]:


# Check head of data
df.head(200)


# In[4]:


df.tail(200)


# # ANALYSIS

# In[5]:


#Check for null values
df.isnull().values.any()


# In[6]:


# Getting information about the data
df.info()


# In[7]:


legit = len(df[df.isFraud == 0])
fraud = len(df[df.isFraud == 1])
legit_percent = (legit / (fraud + legit)) * 100
fraud_percent = (fraud / (fraud + legit)) * 100

print ('Number of Legit transactions: ', legit)
print('Number of Fraud transactions: ', fraud)
print('Percentage of Legit transactions: {:.4f} %'.format(legit_percent))
print('Percentage of Fraud transactions: {:.4f} %'.format(fraud_percent))


# These results prove that this is a highly unbalanced data as Percentage of Legit transactions= 99.87 % and Percentage of Fraud transactions= 0.13 %. SO DECISION TREES AND RANDOM FORESTS ARE GOOD METHODS FOR IMBALANCED DATA.

# In[8]:


# Merchants
X = df[df['nameDest'].str.contains('M')]
X.head()


# For merchants there is no information regarding the attribites oldbalanceDest and newbalanceDest. This is because

# # VISUALISATION

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


pip install seaborn


# For merchants there is no information regarding the attribites oldbalanceDest and newbalanceDest.

# <b>CORRELATION HEATMAP

# In[11]:


corr = df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True)


# NUMBER LEGIT AND FRAUD TRANSACTIONS

# In[12]:


plt.figure(figsize = (5,10))
labels = ['Legit', 'Fraud']
count_classes = df.value_counts(df['isFraud'], sort = True)
count_classes.plot(kind = 'bar', rot = 0)
plt.title('Visualisation of Labels')
plt.ylabel('Count')
plt.xticks(range(2), labels)
plt.show()


# # PROBLEM SOLVING

# In[13]:


# Creating a copy original dataset to train and test models
new_df = df.copy()
new_df.head()


# # LABEL ENCODING
# 

# In[14]:


# Checking how many attributes are dtype:object
objList = new_df.select_dtypes(include = 'object').columns
print(objList)


# THERE ARE 3 ATTRIBUTES WITH Object Datatype. THUS WE NEED TO LABEL ENCODE THEM IN ORDER TO CHECK MULTICOLINEARITY.

# In[15]:


pip install scikit-learn


# In[16]:


# Label Encoding for object to numeric conversion
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    new_df[feat] = le.fit_transform(new_df[feat].astype(str))
    
    print(new_df.info())


# In[17]:


new_df.head()


# # MULTICOLINEARITY

# In[18]:


pip install statsmodels


# In[19]:


#Import Library for VIF (VARIANCE INFLATION FACTOR)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(df):
    #Calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = df.columns
    vif ['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return(vif)
calc_vif(new_df)


# We can see that oldbalanceOrg and newbalanceOrig have too high VIF thus they are highly correlated. Similarly oldbalanceDest and newbalanceDest. Also nameDest is connected to nameOrig.
# 
# Thus combine these pairs of collinear attributes and drop the individual ones.

# In[20]:


new_df['Actual_amount_orig'] = new_df.apply(lambda x: x['oldbalanceOrg'] - x['newbalanceOrig'],axis=1)
new_df['Actual_amount_dest'] = new_df.apply(lambda x: x['oldbalanceDest'] - x['newbalanceDest'],axis=1)
new_df['TransactionPath'] = new_df.apply(lambda x: x['nameOrig'] + x['nameDest'],axis=1)

#Dropping columns
new_df = new_df.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step','nameOrig','nameDest'],axis=1)

calc_vif(new_df)


# In[21]:


corr = new_df.corr()

plt.figure(figsize = (10,6))
sns.heatmap(corr, annot = True)


# How did you select variables to be included in the model?
# Using the VIF values and correlation heatmap. We just need to check if there are any two attributes highly correlated to each other and then drop the one which is less correlated to the isFraud Attribute.

# # MODEL BUILDING

# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# # NOMALIZING (SCALING) AMOUNT

# In[23]:


# Perform Scaling
scaler = StandardScaler()
new_df["NormalizedAmount"] = scaler.fit_transform(new_df["amount"].values.reshape(-1, 1))
new_df.drop(["amount"], inplace= True, axis= 1)

Y = new_df["isFraud"]
X = new_df.drop(["isFraud"], axis= 1)


# I did not normalize the complete dataset because it may lead to decrease in accuracy of model.

# # TRAIN-TEST-SPLIT

# In[24]:


# Split the data
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.2, random_state= 24)


# In[25]:


# DECISION TREE

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred_dt = decision_tree.predict(X_test)
decision_tree_score = decision_tree.score(X_test, Y_test) * 100


# In[ ]:


# RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100


# # EVALUATION

# In[ ]:


# Print Scores of our classifiers 
print('Decision Tree Score: ', decision_tree_score)
print('Random Forest Score: ', random_forest_score)


# In[ ]:


#key terms of confusion matrix- DT
print('TP, FP, TN, FN - Decision Tree')
tn,fp,fn, tp = confusion_matrix(Y_test, Y_pred_dt).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# In[ ]:


#key terms of confusion matrix- RF
print('TP, FP, TN, FN - Random Forest')
tn,fp,fn, tp = confusion_matrix(Y_test, Y_pred_dt).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# * TP(Decision Tree) ~ TP(Random Forest)    No competetion here
# * FP(Decision Tree) >> FP(Random Forest)  Random Forest has an edge
# * TN(Decision Tree) < TN(Random Forest)  Random forest is better here
# * FN(Decision Tree) ~ FN(Random Forest)  No competetion here
# 

# Here, Random Forest looks good

# In[ ]:


#confusion matrix-DT
confusion_matrix_dt = confusion_matrix(Y_test, Y_pred_dt.round())
print('Confusion Matrix - Decision Tree')
print(confusion_matrix_dt)


# In[ ]:


#confusion matrix - RF
confusion_matrix_rf = confusion_matrix(Y_test, Y_pred_rf.round())
print ('Confusion matrix - Random Forest')
print(confusion_matrix_rf)


# In[ ]:


# classification report - DT
classification_report_dt = classification_report(Y_test, Y_pred_dt)
print(classification_report_dt)


# In[ ]:


# classification report - RF
classification_report_rf = classification_report(Y_test, Y_pred_rf)
print('Classification Report - Random Forest')
print(classification_report_rf)


# With such good precision and hence F1 score, Random Forest comes out to be better as expected

# In[ ]:


#visualisation confusion Matrix - Dt
disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_dt)
disp.plot()
plt.title('Confusion Matrix - DT')
plt.show()


# In[ ]:


# visualisation confusion matrix - RF
disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_rf)
disp.plot()
plt.title('Confusion Matrix - Rf')
plt.show()


# In[ ]:


# AUC ROC - DT
# Calculate the fpr and tpr for all thresholds of the classification 

fpr, tpr, threhold = metrics.roc_curve(Y_test, Y_pred_dt)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - DT')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend (loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1]) 
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()         


# In[ ]:


# AUC ROC - RF
# calcualte the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_rf)
roc_auc = metrics.auc(fpr,tpr)

plt.title('ROC-RF')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# AUC for both Decision Tree and Random Forest is eqal, hence both model are good at whta they do 

# # CONCLUSION

# Now we know that the accuracy of both Random Forest and Decision Tree are equal. The accuracy of Random Forest is more comapred to Decision Tree.
# 
# In a fraud detection model, precision is very important because rather than predicting normal transactions correctly we want fraudulent transactions to be predicted correctly and accurately and legit to be left off. If the reasons are not fulfilled, we may catch the innocent and let the culprit free.
# 
# This is one reason why Decision Tree and Random Forest are used over other algorithms.
# 
# The reason of chosen this model is because of highly unbalanced dataset. (Legit:Fraud::99.87:0.13). Random Forest makes multiple decision trees which makes it easier for model to undertand data in simpler way. Decision Tree makes decision in boolean way.
# 
# Model like XGBoost, Bagging, ANN, and Logostic Regression may give good accuracy but wont give good precision and recall values.
# 
# What are key factors taht predict fraudulent customers?
# * The source of request is secured or not.
# * Is the name asking for money legit or not.
# * Transaction history of vendors.
# 
# Prevention policies to be adopted while company updates its infrastructure.
# 
# * Using smart verified apps.
# * Browsing through secured websites.
# * Use secured internet connection (use VPN)
# * Keeping mobile and laptop security updated.
# * Do not respond to unsolicitated calls/SMS/Emails
# * If you feel you have been tricked or the security is compromised, one should contact the bank immediately.
# 
# Assuming these actions have been implemented, how would you determine if they work:
# 
# * Banks sending E-statements.
# * Customers keeping a check on their accounta activity
# * Always keep a log of payments.
