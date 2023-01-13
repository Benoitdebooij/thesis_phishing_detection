#!/usr/bin/env python
# coding: utf-8

# In[162]:


import pandas as pd
import numpy as np
import random as rn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from numpy import mean
from numpy import std
import warnings
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[31]:


np.random.seed(20)


# In[10]:


data = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\Legit_legit_data\final_dataset_v3.csv')
data.drop(data.filter(regex="Unname"),axis=1, inplace=True)
data


# Performing EDA and dataset description

# In[11]:


#label count phishing
data['Label'].value_counts()[1]


# In[12]:


#label count legit 
data['Label'].value_counts()[0]


# In[13]:


#Plotting the feature distribution
data.hist(bins = 10,figsize = (15,15))


# In[14]:


#Plotting the feature distribution
data["Email"].hist(bins = 10,figsize = (4,4))
plt.savefig('features_distribution_email.png')


# In[15]:


#Plotting the data distribution
data["Anchor"].hist(bins = 10,figsize = (4,4))
plt.savefig('features_distribution_anchor.png')


# In[16]:


#removing useless features
del data["IPAdress"]
del data["Email"]


# In[17]:


data.describe()


# In[18]:


#drop label for X and drop features for Y
Y = data['Label']  #target variable
X = data.drop('Label',axis=1)   #independent variable


# Splitting data in training, validation and test set

# In[19]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[20]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
    test_size=0.2)


# In[21]:


x_train.shape
y_train.shape


# In[22]:


x_val.shape
y_val.shape


# In[23]:


x_test.shape
y_test.shape


# Logistic Regression

# In[24]:


#baseline model
lr_model = LogisticRegression()


# In[25]:


lr_model.get_params()


# In[26]:


#fitting model and making predictions
cv = RepeatedKFold(n_splits=10, n_repeats=10)
lr_model.fit(x_train, y_train)
predictions_lr = lr_model.predict(x_test)


# In[27]:


#10 fold cross validation
f1_lr_cv = cross_val_score(lr_model, x_test, y_test, cv=cv, scoring='f1')
f1_score_lr = sum(f1_lr_cv)/len(f1_lr_cv)
recall_lr_cv = cross_val_score(lr_model, x_test, y_test, cv=cv, scoring='recall')
recall_score_lr= sum(recall_lr_cv)/len(recall_lr_cv)
precision_lr_cv = cross_val_score(lr_model, x_test, y_test, cv=cv, scoring='precision')
precision_score_lr= sum(precision_lr_cv)/len(precision_lr_cv)


# In[28]:


#standard deviations
std_f1_lr = std(f1_lr_cv)
std_recall_lr = std(recall_lr_cv)
std_precision_lr = std(precision_lr_cv)


# In[29]:


ML_model = ["Logistic Regression"]
std_lr = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_lr,
                        'Recall': std_recall_lr,
                        "Precision" : std_precision_lr})
std_lr


# In[30]:


ML_model = ["Logistic Regression"]
results_lr = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_lr,
                        'Recall': recall_score_lr,
                        "Precision" : precision_score_lr})
results_lr


# In[32]:


#plotting confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=predictions_lr)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')
ax.set_title('Logistic Regression confustion matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()


# In[34]:


#Misclassification Rate
mis_clas_lr = (fp + fn)/(fp+tp+fn+tn)
# False Positive Rate
FPR_lr = fp/(fp+tn)
#Specificity
TNR_lr = tn/(tn+fp)
#False Negative Ratio
FNR_lr = fn/(tp+fn)
#False discovery rate
FDR_lr = fp/(tp+fp)


# Random Forest

# In[35]:


rf = RandomForestClassifier()


# In[36]:


RandomForestClassifier().get_params()


# In[37]:


#creating lists with ranges for randomizedsearch hyperparameter tuning
random_rf = {
    'n_estimators': list(range(10,200,50)),
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(1,41,10)),
    'min_samples_split': list(range(1, 41, 10)),
    'min_samples_leaf': list(range(1, 41, 10)),
    'max_features': ['sqrt', 'log2'],
    'bootstrap' : [True, False]
}


# In[38]:


#performing randomizedsearch
random_grid_rf = RandomizedSearchCV(estimator=rf,
                              param_distributions=random_rf,
                              n_iter=50,
                              cv=5,
                              scoring="f1",
                              return_train_score=True,
                              verbose=2)
random_grid_rf.fit(x_val,y_val)


# In[39]:


#returning best params according to randomized search
random_grid_rf.best_params_


# In[48]:


#more specific range of parameters for gridsearch 
param_grid_rf = {
    'n_estimators': list(range(155,170,5)),
    'criterion': ['gini'],
    'max_depth': list(range(18,24,1)),
    'min_samples_split': list(range(18, 24, 1)),
    'min_samples_leaf': [0.8,1,1.2, 2],
    'max_features': ['log2'],
    'bootstrap' : [False]
}


# In[47]:


param_grid_rf_2 = pd.DataFrame(grid_search_rf)


# In[50]:


#performing gridsearch
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search_rf.fit(x_val, y_val)


# In[51]:


#returning best parameters after gridsearch
grid_search_rf.best_params_


# In[63]:


#the optimized model 
rf_model = RandomForestClassifier(criterion='gini',
                                      max_depth=18,
                                      max_features='log2',
                                      min_samples_leaf=1,
                                      min_samples_split=18,
                                      n_estimators=155,
                                      bootstrap=False)


# In[60]:


#training the model and making predictions
cv = RepeatedKFold(n_splits=10, n_repeats=10)
rf_model.fit(x_train, y_train)
predictions_rf = rf_model.predict(x_test)


# In[61]:


#10 fold cross validation
f1_rf_cv = cross_val_score(rf_model, x_test, y_test, cv=cv, scoring='f1')
f1_score_rf = sum(f1_rf_cv)/len(f1_rf_cv)
recall_rf_cv = cross_val_score(rf_model, x_test, y_test, cv=cv, scoring='recall')
recall_score_rf= sum(recall_rf_cv)/len(recall_rf_cv)
precision_rf_cv = cross_val_score(rf_model, x_test, y_test, cv=cv, scoring='precision')
precision_score_rf= sum(precision_rf_cv)/len(precision_rf_cv)


# In[62]:


#standard deviations
std_f1_rf = std(f1_rf_cv)
std_recall_rf = std(recall_rf_cv)
std_precision_rf = std(precision_rf_cv)


# In[56]:


ML_model = ["Random Forest"]
std_rf = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_rf,
                        'Recall': std_recall_rf,
                        "Precision" : std_precision_rf})
std_rf


# In[57]:


ML_model = ["Random Forest"]
results_rf = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_rf,
                        'Recall': recall_score_rf,
                        "Precision" : precision_score_rf})
results_rf


# In[58]:


cm = confusion_matrix(y_true=y_test, y_pred=predictions_rf)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')

ax.set_title('Random Forest confustion matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[64]:


#Misclassification Rate
mis_clas_rf = (fp + fn)/(fp+tp+fn+tn)
# False positive rate
FPR_rf = fp/(fp+tn)
#Specificity
TNR_rf = (tn)/(tn+fp)
#False negative rate
FNR_rf = fn/(tp+fn)
#False discovery rate
FDR_rf = fp/(tp+fp)


# KNN

# In[65]:


knn = KNeighborsClassifier()


# In[66]:


KNeighborsClassifier().get_params()


# In[78]:


#Parameter ranges which are tested by the RandomizedSearch
random_knn = {'leaf_size' : list(range(1,50)),
'metric' : ['minkowski', 'manhattan', 'euclidean'],
'p' : [1, 2],
'n_neighbors' : list(range(1,30)),
'weights' : ['uniform', 'distance'],
'n_jobs' : list(range(1,50))
}


# In[79]:


#Performing RandomizedSearch
random_grid_knn = RandomizedSearchCV(estimator=knn,
                              param_distributions=random_knn,
                              n_iter=50,
                              cv=5,
                              scoring="f1",
                              return_train_score=True,
                              verbose=2)
random_grid_knn.fit(x_val,y_val)


# In[69]:


random_grid_knn.best_params_


# In[77]:


#Param grid after RandomizedSearch
param_grid_knn = {'leaf_size' : list(range(38,45,1)),
'metric' : ['manhattan'],
'p' : [1],
'n_neighbors' : list(range(6,13)),
'weights' : ['distance'],
'n_jobs' : list(range(38,50,1))
}


# In[71]:


#Performing Grid Search
grid_search_knn = GridSearchCV(estimator = knn, param_grid = param_grid_knn, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search_knn.fit(x_val, y_val)


# In[76]:


#Grid search best parameters
grid_search_knn.best_params_


# In[84]:


#KNN model final
knn_model = KNeighborsClassifier(leaf_size= 38,
 metric='manhattan',
 n_neighbors=10,
 p= 1,
 weights='distance',
 n_jobs= 38)


# In[83]:


#Fitting the KNN and make predictions
cv = RepeatedKFold(n_splits=10, n_repeats=10)
knn_model.fit(x_train, y_train)
predictions_knn = knn_model.predict(x_test)


# In[85]:


#10 fold cross validation
f1_knn_cv = cross_val_score(knn_model, x_test, y_test, cv=cv, scoring='f1')
f1_score_knn = sum(f1_knn_cv)/len(f1_knn_cv)
recall_knn_cv = cross_val_score(knn_model, x_test, y_test, cv=cv, scoring='recall')
recall_score_knn= sum(recall_knn_cv)/len(recall_knn_cv)
precision_knn_cv = cross_val_score(knn_model, x_test, y_test, cv=cv, scoring='precision')
precision_score_knn= sum(precision_knn_cv)/len(precision_knn_cv)
balanced_acc_knn_cv = cross_val_score(knn_model, x_test, y_test, cv=cv, scoring='balanced_accuracy')
balanced_acc_score_knn= sum(balanced_acc_knn_cv)/len(balanced_acc_knn_cv)


# In[86]:


#Standard Deviations
std_f1_knn = std(f1_knn_cv)
std_recall_knn = std(recall_knn_cv)
std_precision_knn = std(precision_knn_cv)


# In[87]:


ML_model = ["KNN"]
std_knn = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_knn,
                        'Recall': std_recall_knn,
                        "Precision" : std_precision_knn})
std_knn


# In[88]:


ML_model = ["K-nearest neighbors"]
#creating dataframe
results_knn = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_knn,
                        'Recall': recall_score_knn,
                        "Precision" : precision_score_knn})

results_knn


# In[89]:


cm = confusion_matrix(y_true=y_test, y_pred=predictions_knn)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')

ax.set_title('KNN confustion matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[91]:


#Misclassification Rate
mis_clas_knn = (fp + fn)/(fp+tp+fn+tn)
# False positive rate
FPR_knn = fp/(fp+tn)
#Specificity
TNR_knn = tn/(tn+fp)
#False negative rate
FNR_knn = fn/(tp+fn)
#False discovery rate
FDR_knn = fp/(tp+fp)


# SVM model

# In[92]:


svm_standard = svm.SVC()


# In[93]:


svm_standard.get_params()


# In[94]:


#Parameter ranges which are tested by the RandomizedSearch
random_svm = {'C': list(range(1,200, 10)), 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'polynomial','rbf', 'sigmoid'],
              'degree' : list(range(1,100,10))
}


# In[95]:


#Performing RandomizedSearch
random_grid_svm = RandomizedSearchCV(estimator=svm_standard,
                              param_distributions=random_svm,
                              n_iter=50,
                              cv=5,
                              scoring="f1",
                              return_train_score=True,
                              verbose=2)
random_grid_svm.fit(x_val,y_val)


# In[97]:


#Showing best parameters
random_grid_svm.best_params_


# In[99]:


#Param grid before Grid Search
param_grid_svm = {'C': list(range(1,90, 5)), 
              'gamma': [0.01],
              'kernel': ['rbf'],
              'degree' : list(range(1,34,2))
}


# In[100]:


grid_search_svm = GridSearchCV(estimator = svm_standard, param_grid = param_grid_svm, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search_svm.fit(x_val, y_val)


# In[103]:


#Showing best params after Grid Search
grid_search_svm.best_params_


# In[104]:


#Fitting the SVM and make predictions
svm_model = svm.SVC(C=66, degree=1, gamma=0.01, kernel='rbf', probability=True)
svm_model.fit(x_train, y_train)
predictions_svm = svm_model.predict(x_test)


# In[106]:


#10 fold cross validation
f1_svm_cv = cross_val_score(svm_model, x_test, y_test, cv=cv, scoring='f1')
f1_score_svm = sum(f1_svm_cv)/len(f1_svm_cv)
recall_svm_cv = cross_val_score(svm_model, x_test, y_test, cv=cv, scoring='recall')
recall_score_svm= sum(recall_svm_cv)/len(recall_svm_cv)
precision_svm_cv = cross_val_score(svm_model, x_test, y_test, cv=cv, scoring='precision')
precision_score_svm= sum(precision_svm_cv)/len(precision_svm_cv)


# In[107]:


#Standard deviations
std_f1_svm = std(f1_svm_cv)
std_recall_svm = std(recall_svm_cv)
std_precision_svm = std(precision_svm_cv)


# In[108]:


ML_model = ["SVM"]
std_svm = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_svm,
                        'Recall': std_recall_svm,
                        "Precision" : std_precision_svm})
std_svm


# In[109]:


ML_model = ["SVM"]
#creating dataframe
results_svm = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_svm,
                        'Recall': recall_score_svm,
                        "Precision" : precision_score_svm})

results_svm


# In[112]:


#Plotting confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=predictions_svm)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')

ax.set_title('Support Vector Machine confustion matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[111]:


#Misclassification Rate
mis_clas_svm = (fp + fn)/(fp+tp+fn+tn)
# False positive rate
FPR_svm = fp/(fp+tn)
#Specificity
TNR_svm = tn/(tn+fp)
#False negative rate
FNR_svm = fn/(tp+fn)
#False discovery rate
FDR_svm = fp/(tp+fp)


# XGBoost

# In[113]:


xgb = XGBClassifier()


# In[115]:


#Get params
xgb.get_params()


# In[116]:


#Range of parameters which are tested in RandomizedSearch
random_xgb = {
        'min_child_weight': list(range(1, 40, 4)),
        'gamma': list(range(1, 16, 4)),
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': list(range(1, 40, 4))
        }


# In[117]:


#Performing RandomizedSearch
random_grid_xgb = RandomizedSearchCV(estimator=xgb,
                              param_distributions=random_xgb,
                              n_iter=50,
                              cv=5,
                              scoring="f1",
                              return_train_score=True,
                              verbose=2)
random_grid_xgb.fit(x_val,y_val)


# In[118]:


#Best params after RandomizedSearch
random_grid_xgb.best_params_


# In[120]:


#Params which are used in the Grid Search
param_grid_xgb = {
        'min_child_weight': [1.2, 1.3],
        'gamma': list(range(5, 7, 1)),
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.8, 0.9, 1.1],
        'max_depth': list(range(6, 10, 1))
        }


# In[121]:


#Perform Grid Search
grid_search_xgb = GridSearchCV(estimator = xgb, param_grid = param_grid_xgb, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search_xgb.fit(x_val, y_val)


# In[122]:


#Best params after GridSearch
grid_search_xgb.best_params_


# In[124]:


#XGBoost model
xgb_model = XGBClassifier(colsample_bytree=0.9, gamma=6, max_depth=5, min_child_weight=3, subsample=0.9)


# In[125]:


#Fit the model and make predictions
cv = RepeatedKFold(n_splits=10, n_repeats=10)
xgb_model.fit(x_train, y_train)
predictions_xgb = xgb_model.predict(x_test)


# In[126]:


#10 fold cross validation
f1_xgb_cv = cross_val_score(xgb_model, x_test, y_test, cv=cv, scoring='f1')
f1_score_xgb= sum(f1_xgb_cv)/len(f1_xgb_cv)
recall_xgb_cv = cross_val_score(xgb_model, x_test, y_test, cv=cv, scoring='recall')
recall_score_xgb= sum(recall_xgb_cv)/len(recall_xgb_cv)
precision_xgb_cv = cross_val_score(xgb_model, x_test, y_test, cv=cv, scoring='precision')
precision_score_xgb= sum(precision_xgb_cv)/len(precision_xgb_cv)


# In[127]:


#Standard deviations
std_f1_xgb = std(f1_xgb_cv)
std_recall_xgb = std(recall_xgb_cv)
std_precision_xgb = std(precision_xgb_cv)


# In[128]:


ML_model = ["XGB"]
std_xgb = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_xgb,
                        'Recall': std_recall_xgb,
                        "Precision" : std_precision_xgb})
std_xgb


# In[129]:


ML_model = ["XGB"]
results_xgb = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_xgb,
                        'Recall': recall_score_xgb,
                        "Precision" : precision_score_xgb})

results_xgb


# In[130]:


#Plotting confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=predictions_xgb)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')

ax.set_title('XGBoost confustion matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[131]:


#Misclassification Rate
mis_clas_xgb = (fp + fn)/(fp+tp+fn+tn)
# False positive rate
FPR_xgb = fp/(fp+tn)
#Specificity
TNR_xgb = tn/(tn+fp)
#False negative rate
FNR_xgb = fn/(tp+fn)
#False discovery rate
FDR_xgb = fp/(tp+fp)


# Stacking Model

# In[133]:


#Assembling Stacked model
def get_stacking_model():
    # define the base models
    level0 = []
    
    level0.append(['KNN', Pipeline([('classifier', knn_model)])])
    
    level0.append(['SVM', Pipeline([('classifier', svm_model)])])

    level0.append(['Random Forest', Pipeline([('classifier', rf_model)])])
    level1 = xgb_model
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10)
    return model


# In[134]:


#Fit Stacked model and make predictions
model_stacked = get_stacking_model()
cv = RepeatedKFold(n_splits=10, n_repeats=10)
model_stacked.fit(x_train, y_train)
predictions_stacked = model_stacked.predict(x_test)


# In[135]:


#10 fold cross validation
f1_model = cross_val_score(model_stacked, x_test, y_test, cv=cv, scoring='f1')
f1_score_model= sum(f1_model)/len(f1_model)
recall_model_cv = cross_val_score(model_stacked, x_test, y_test, cv=cv, scoring='recall')
recall_score_model= sum(recall_model_cv)/len(recall_model_cv)
precision_model_cv = cross_val_score(model_stacked, x_test, y_test, cv=cv, scoring='precision')
precision_score_model= sum(precision_model_cv)/len(precision_model_cv)


# In[137]:


#Standard deviation
std_f1_stacked = std(f1_model)
std_recall_stacked = std(recall_model_cv)
std_precision_stacked = std(precision_model_cv)


# In[138]:


ML_model = ["Stacked Model"]
std_stacked = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_stacked,
                        'Recall': std_recall_stacked,
                        "Precision" : std_precision_stacked})
std_stacked


# In[139]:


ML_model = ["Stacked Model"]
#creating dataframe
results_model = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_model,
                        'Recall': recall_score_model,
                        "Precision" : precision_score_model})

results_model


# In[140]:


#Plotting confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=predictions_stacked)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')

ax.set_title('Stacked Model confustion matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[141]:


#Misclassification Rate
mis_clas_stacked = (fp + fn)/(fp+tp+fn+tn)
# False positive rate
FPR_stacked = fp/(fp+tn)
#Specificity
TNR_stacked = tn/(tn+fp)
#False negative rate
FNR_stacked = fn/(tp+fn)
#False discovery rate
FDR_stacked = fp/(tp+fp)


# Bagging classifier

# In[144]:


# bagging classifier before tuning
bag_model_tuning = BaggingClassifier()


# In[145]:


#Get parameters of bagging model
bag_model_tuning.get_params()


# In[147]:


#Function to print f1 score of training and validation set with the different n_estimators in the list
n_estimators = [1, 10, 25, 50, 75, 100, 200, 500, 1000]

for n_estimators in n_estimators:
    bagging_model_tuning = BaggingClassifier(n_estimators=n_estimators)
    bagging_model_tuning.fit(x_train, y_train)

    print("n_estimators: ", n_estimators)
    print("Training", bagging_model_tuning.score(x_train, y_train))
    print("Validation", bagging_model_tuning.score(x_val, y_val))


# In[148]:


#Function to print f1 score of training and validation set with the different max_features in the list
max_features = [1, 10, 20]

for max_features in max_features:
    boosting_model_tuning = BaggingClassifier(n_estimators=75, max_features=max_features)
    boosting_model_tuning.fit(x_train, y_train)

    print("n_estimators: ", max_features)
    print("Training", boosting_model_tuning.score(x_train, y_train))
    print("Validation", boosting_model_tuning.score(x_val, y_val))


# In[149]:


#Optimized Bagging model
bag_model = BaggingClassifier(n_estimators=75, max_features=20)


# In[150]:


#Fitting the model and making predictions
bag_model.fit(x_train, y_train)
predictions_bag = bag_model.predict(x_test)
cv = RepeatedKFold(n_splits=10, n_repeats=10)


# In[151]:


#10 fold cross validation
f1_bag_model = cross_val_score(bag_model, x_test, y_test, cv=cv, scoring='f1')
f1_score_bag_model= sum(f1_bag_model)/len(f1_bag_model)
recall_bag_model_cv = cross_val_score(bag_model, x_test, y_test, cv=cv, scoring='recall')
recall_score_bag_model= sum(recall_bag_model_cv)/len(recall_bag_model_cv)
precision_bag_model_cv = cross_val_score(bag_model, x_test, y_test, cv=cv, scoring='precision')
precision_score_bag_model= sum(precision_bag_model_cv)/len(precision_bag_model_cv)


# In[157]:


#Stamdard deviations
std_f1_bag = std(f1_bag_model)
std_recall_bag = std(recall_bag_model_cv)
std_precision_bag = std(precision_bag_model_cv)


# In[155]:


ML_model = ["Bagging"]
results_bagging = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_bag_model,
                        'Recall': recall_score_bag_model,
                        "Precision" : precision_score_bag_model})

results_bagging


# In[158]:


ML_model = ["Bagging Model"]
std_bag = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_bag,
                        'Recall': std_recall_bag,
                        "Precision" : std_precision_bag})
std_bag


# In[156]:


#Plotting confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=predictions_bag)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')
ax.set_title('Bagging Model confustion matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()


# In[159]:


#Misclassification Rate
mis_clas_bag = (fp + fn)/(fp+tp+fn+tn)
# False positive rate
FPR_bag = fp/(fp+tn)
#Specificity
TNR_bag = tn/(tn+fp)
#False negative rate
FNR_bag = fn/(tp+fn)
#False discovery rate
FDR_bag = fp/(tp+fp)


# Boosting model

# In[163]:


boosting = GradientBoostingClassifier()


# In[164]:


#Get parameters
boosting.get_params()


# In[165]:


#Function to print f1 score of training and validation set with the different max_depth values in the list
max_depth = [1, 2, 4, 6, 8, 10, 12, 14, 16]

for max_depth in max_depth:
    boosting_model_tuning = GradientBoostingClassifier(max_depth=max_depth)
    boosting_model_tuning.fit(x_train, y_train)

    print("max_depth: ", max_depth)
    print("Training", boosting_model_tuning.score(x_train, y_train))
    print("Validation", boosting_model_tuning.score(x_val, y_val))


# In[166]:


#Function to print f1 score of training and validation set with the different learning_rate values in the list
learning_list = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5 ]

for learning_rate in learning_list:
    boosting_model_tuning = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=10)
    boosting_model_tuning.fit(x_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Training", boosting_model_tuning.score(x_train, y_train))
    print("Validation", boosting_model_tuning.score(x_val, y_val))


# In[167]:


#Function to print f1 score of training and validation set with the different n_estimators values in the list
n_estimators = [1, 10, 25, 50, 75, 100, 200, 500, 1000]

for n_estimators in n_estimators:
    boosting_model_tuning = GradientBoostingClassifier(learning_rate=0.25, max_depth=10, n_estimators=n_estimators)
    boosting_model_tuning.fit(x_train, y_train)

    print("n_estimators: ", n_estimators)
    print("Training", boosting_model_tuning.score(x_train, y_train))
    print("Validation", boosting_model_tuning.score(x_val, y_val))


# In[168]:


#Optimized Boosting model
boosting_model = GradientBoostingClassifier(learning_rate=0.25, max_depth=10, n_estimators=100)


# In[169]:


#Fitting the model and making predictions
boosting_model.fit(x_train, y_train)
predictions_boosting = boosting_model.predict(x_test)
cv = RepeatedKFold(n_splits=10, n_repeats=10)


# In[170]:


#10 fold cross validation
f1_boosting = cross_val_score(boosting_model, x_test, y_test, cv=cv, scoring='f1')
f1_score_boosting= sum(f1_boosting)/len(f1_boosting)
recall_boosting_cv = cross_val_score(boosting_model, x_test, y_test, cv=cv, scoring='recall')
recall_score_boosting= sum(recall_boosting_cv)/len(recall_boosting_cv)
precision_boosting_cv = cross_val_score(boosting_model, x_test, y_test, cv=cv, scoring='precision')
precision_score_boosting= sum(precision_boosting_cv)/len(precision_boosting_cv)


# In[173]:


#Standard deviation
std_f1_boosting = std(f1_boosting)
std_recall_boosting = std(recall_boosting_cv)
std_precision_boosting = std(precision_boosting_cv)


# In[172]:


ML_model = ["Boosting Model"]
results_boosting = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': f1_score_boosting,
                        'Recall': recall_score_boosting,
                        "Precision" : precision_score_boosting})

results_boosting


# In[174]:


ML_model = ["Boosting Model"]
std_boosting = pd.DataFrame({ 'ML Model': ML_model,    
                         'F1 score': std_f1_boosting,
                        'Recall': std_recall_boosting,
                        "Precision" : std_precision_boosting})
std_boosting


# In[175]:


#Plotting confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=predictions_boosting)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt = "0.1f", cmap='Blues')

ax.set_title('Boosting Model confustion matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[176]:


#Misclassification Rate
mis_clas_boosting = (fp + fn)/(fp+tp+fn+tn)
# False positive rate
FPR_boosting = fp/(fp+tn)
#Specificity
TNR_boosting = tn/(tn+fp)
#False negative rate
FNR_boosting = fn/(tp+fn)
#False discovery rate
FDR_boosting = fp/(tp+fp)


# Results

# In[179]:


#Merge result dataframes
frames = [results_lr, results_rf, results_knn, results_svm, results_xgb, results_bagging, results_boosting, results_model]
final_results = pd.concat(frames)


# In[178]:


#Plotting results in bar chart
final_results2 = final_results.round(2)
labels = ["LR", "RF", "KNN", "SVM", "XGB", "Bagging", "Boosting", "Stacking"]
N=8
ind = np.arange(N)
ax = final_results2.plot.bar(figsize=(13, 4), align='edge', width=0.8)
ax.bar_label(ax.containers[0], label_type='edge')
ax.bar_label(ax.containers[1], label_type='edge')
ax.bar_label(ax.containers[2], label_type='edge')
plt.legend(loc='lower left')
ax.set_xticks(ind, labels)
ax.set_xticklabels(labels, rotation=0 )


# In[180]:


#Merge MR 
Measure = ["Misclassification Rate"]
results_mis_clas = pd.DataFrame({ 'Measure': Measure,    
                         'Logistic Regression': mis_clas_lr,
                        'Random Forest': mis_clas_rf,
                        "SVM" : mis_clas_lr,
                        "KNN" : mis_clas_knn,
                        "XGBoost" : mis_clas_xgb,
                        "Stacked Model" : mis_clas_stacked,
                        "Bagging": mis_clas_bag,
                        "Boosting": mis_clas_boosting})
results_mis_clas


# In[181]:


#Merge FPR
Measure = ["False Positive Rate"]
results_FPR = pd.DataFrame({ 'Measure': Measure,    
                         'Logistic Regression': FPR_lr,
                        'Random Forest': FPR_rf,
                        "SVM" : FPR_svm,
                        "KNN" : FPR_knn,
                        "XGBoost" : FPR_xgb,
                        "Stacked Model" : FPR_stacked,
                        "Bagging": FPR_bag,
                        "Boosting": FPR_boosting})
results_FPR


# In[182]:


#Merge Specificity
Measure = ["Specificity"]
results_TNR = pd.DataFrame({ 'Measure': Measure,    
                         'Logistic Regression': TNR_lr,
                        'Random Forest': TNR_rf,
                        "SVM" : TNR_svm,
                        "KNN" : TNR_knn,
                        "XGBoost" : TNR_xgb,
                        "Stacked Model" : TNR_stacked,
                        "Bagging": TNR_bag,
                        "Boosting": TNR_boosting})
results_TNR


# In[183]:


#Merge FNR
Measure = ["False Negative Rate"]
results_FNR = pd.DataFrame({ 'Measure': Measure,    
                         'Logistic Regression': FNR_lr,
                        'Random Forest': FNR_rf,
                        "SVM" : FNR_svm,
                        "KNN" : FNR_knn,
                        "XGBoost" : FNR_xgb,
                        "Stacked Model" : FNR_stacked,
                        "Bagging": FNR_bag,
                        "Boosting": FNR_boosting})
results_FNR


# In[184]:


#Merge FDR
Measure = ["False Discovery Rate"]
results_FDR = pd.DataFrame({ 'Measure': Measure,    
                         'Logistic Regression': FDR_lr,
                        'Random Forest': FDR_rf,
                        "SVM" : FDR_svm,
                        "KNN" : FDR_knn,
                        "XGBoost" : FDR_xgb,
                        "Stacked Model" : FDR_stacked,
                        "Bagging": FDR_bag,
                        "Boosting": FDR_boosting})
results_FDR


# In[185]:


#Merge standard deviations
frames_std = [std_lr, std_rf, std_knn, std_svm, std_xgb, std_stacked, std_bag, std_boosting]
final_results_std = pd.concat(frames_std)
final_results_std


# In[186]:


#Merge additional metrics in one dataframe
frames_metrics = [results_FDR, results_FNR, results_FPR, results_TNR, results_mis_clas]
final_metrics = pd.concat(frames_metrics)
final_metrics


# ROC curve and AUC scores

# In[187]:


#Plot ROC
y_pred_proba_lr = lr_model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_lr)
auc_lr = metrics.roc_auc_score(y_test, y_pred_proba_lr)
#create ROC curve
plt.title('ROC Curve Logistic Regression')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[188]:


y_pred_proba_rf = rf_model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba_rf)
#create ROC curve
plt.title('ROC Curve Random Forest')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[189]:


y_pred_proba_knn = knn_model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_knn)
auc_knn = metrics.roc_auc_score(y_test, y_pred_proba_knn)
#create ROC curve
plt.title('ROC Curve KNN')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[190]:


y_pred_proba_svm = svm_model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_svm)
auc_svm = metrics.roc_auc_score(y_test, y_pred_proba_svm)
#create ROC curve
plt.title('ROC Curve SVM')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[191]:


y_pred_proba_xgb = xgb_model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_xgb)
auc_xgb = metrics.roc_auc_score(y_test, y_pred_proba_xgb)
#create ROC curve
plt.title('ROC Curve XGBoost')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[192]:


y_pred_proba_stacked = model_stacked.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_stacked)
auc_stacked = metrics.roc_auc_score(y_test, y_pred_proba_stacked)
#create ROC curve
plt.title('ROC Curve Stacked model')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[193]:


y_pred_proba_bag = bag_model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_bag)
auc_bag = metrics.roc_auc_score(y_test, y_pred_proba_bag)
#create ROC curve
plt.title('ROC Curve Bagging model')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[194]:


y_pred_proba_boost = boosting_model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_boost)
auc_boosting = metrics.roc_auc_score(y_test, y_pred_proba_boost)
#create ROC curve
plt.title('ROC Curve Boosting model')
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[195]:


#Merge AUC scores in a single dataframe
Measure = ["AUC"]
results_auc = pd.DataFrame({ 'Measure': Measure,    
                         'Logistic Regression': auc_lr,
                        'Random Forest': auc_rf,
                        "SVM" : auc_svm,
                        "KNN" : auc_knn,
                        "XGBoost" : auc_xgb,
                        "Stacked Model" : auc_stacked,
                        "Bagging Model": auc_bag,
                        "Boosting Model": auc_boosting})
results_auc


# Correlation matrix

# In[197]:


#Plot correlation matrix features
correlation_mat =data.corr()
plt.figure(figsize=(9,9))
sns.heatmap(correlation_mat, annot = False)
plt.title("Correlation matrix of the features")
plt.show()


# Error Analysis

# In[198]:


data_fn = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\test_predictions4.csv')
data_fn.drop(data_fn.filter(regex="Unname"),axis=1, inplace=True)
data_fn


# In[199]:


pred_stacked = model_stacked.predict(data_fn)
pred_stacked


# In[200]:


pred_svm = svm_model.predict(data_fn)
pred_svm


# In[ ]:




