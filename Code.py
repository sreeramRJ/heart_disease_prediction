#!/usr/bin/env python
# coding: utf-8
#Libraries for INstallation (If not already available)
#pip install scikit-learn
#pip install Keras
#pip install eli5#imports
import pandas as pd
import glob

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

def read_data():
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',header=None,names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'])
    return data

#data.head()

#understanding the result
data['num'].unique()

#change result variable to binary as mentioned in dataset description
# In[45]:


data.loc[data['num'] > 0,'num'] = 1

#check for NA
# In[7]:


data.isnull().sum()

#check for the datatypes for any unusual characters
# In[8]:


data.dtypes


# In[9]:


print(data['ca'].unique())
print(data['thal'].unique())


# In[10]:


data.loc[(data['ca'] == '?') | (data['thal'] == '?')]

#replace ca with 0 as it is in most other data points
#replace thal corresponding to the result (num) in other data points - so the prediction model can be effective in training
# In[46]:


data.loc[data['ca'] == '?','ca'] = 0.0
data.loc[(data['thal'] == '?') & (data['num'] == 0),'thal'] = 3.0
data.loc[(data['thal'] == '?') & (data['num'] == 1),'thal'] = 7.0


# In[47]:


data = data.astype({'ca' : 'float64','thal' : 'float64'})


# In[ ]:


#understanding the dataset


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[31]:


plt.rcParams["figure.figsize"] = (12,12)
plt.subplots_adjust(top = 1.5,hspace = 3.0)


# In[32]:


data.hist()


# In[39]:


f = sns.countplot(x='num', data=data)
f.set_title("Heart disease distribution")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.rcParams["figure.figsize"] = (6,6)
plt.xlabel("");


# In[64]:


f = sns.countplot(x='num', data=data, hue='sex')
plt.legend(['Female', 'Male'])
f.set_title("Heart disease by gender")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.xlabel("");


# In[41]:


pd.crosstab(data['age'],data['num']).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
# plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[42]:


from sklearn.preprocessing import StandardScaler
dataset = pd.get_dummies(data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])


# In[48]:


data.head()


# In[103]:


plt.scatter(x=data.age[data.num==1], y=data.thalach[(data.num==1)], c="blue", s=80)
plt.scatter(x=data.age[data.num==0], y=data.thalach[(data.num==0)], s=80)
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("thalach");


# In[50]:


################################## data preprocessing
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[52]:


################################## hyperparameter tuning with Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced',random_state=42)
param_grid = { 
    'C': [0.1,0.2,0.3,0.4],
    'penalty': ['l1', 'l2'],
    'class_weight':[{0: 1, 1: 1},{ 0:0.67, 1:0.33 },{ 0:0.75, 1:0.25 },{ 0:0.8, 1:0.2 }]}
CV_rfc = GridSearchCV(estimator=lr, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_


# In[53]:


#########################################   Logistic Regression  #############################################################

classifier_logistic = LogisticRegression(C=0.4,random_state=42,penalty='l2',class_weight={0: 0.67, 1: 0.33})
classifier_logistic.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_logistic.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier_logistic.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)
sensitivity2 = cm_train[1,1]/(cm_train[1,1]+cm_train[1,0])
print('Sensitivity/Recall : ', sensitivity2)

specificity2 = cm_train[0,0]/(cm_train[0,0]+cm_train[0,1])
print('Specificity : ', specificity2)

precision2 = cm_train[1,1]/(cm_train[1,1]+cm_train[0,1])
print('Precision   : ', precision2)

F1score2=(2*sensitivity2*precision2)/(sensitivity2+precision2)
print('F1 score    : ', F1score2)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


# In[54]:


################################## k fold cross validation
from sklearn.model_selection import StratifiedKFold,cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(X_train, y_train)
results = cross_val_score(classifier_logistic, X_train, y_train, cv=skf, n_jobs=1, scoring='accuracy')
results.mean()


# In[148]:


X_test


# In[55]:


################################## feature importance
import eli5
from eli5.sklearn import PermutationImportance
perm_imp_lr = PermutationImportance(classifier_logistic, random_state=42,scoring='accuracy').fit(X_test, y_test)
eli5.show_weights(perm_imp_lr, feature_names = data.iloc[:,:-1].columns.tolist(),top=50)


# In[56]:


import seaborn as sns
sns.regplot(x='thal', y='num', data=data, logistic=True)


# In[57]:


sns.regplot(x='trestbps', y='num', data=data, logistic=True)


# In[60]:


################################## hyperparameter tuning with Grid Search
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rand_forest = RandomForestClassifier(random_state=42)

rf_random = RandomizedSearchCV(estimator=rand_forest, param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
rf_random.best_params_


# In[61]:


#########################################  Random Forest  #############################################################

classifier_rf = RandomForestClassifier(n_estimators= 2000,
 min_samples_split = 5,
 min_samples_leaf = 1,
 max_features = 'sqrt',
 max_depth = 10,
 bootstrap = True)
classifier_rf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_rf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier_rf.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


# In[203]:


y_pred


# In[62]:


perm_imp_rf = PermutationImportance(classifier_rf, random_state=42,scoring='accuracy').fit(X_test, y_test)
eli5.show_weights(perm_imp_rf, feature_names = data.iloc[:,:-1].columns.tolist(),top=50)


# In[63]:


################################## k fold cross validation
results = cross_val_score(classifier_rf, X_train, y_train, cv=skf, n_jobs=1, scoring='accuracy')
results.mean()


# In[65]:


###############################################################################
# applying XGBoost

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 0)

from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = xg.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i]>= 0.5:       # setting threshold to .5
       y_pred_train[i]=1
    else:  
       y_pred_train[i]=0
       
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


# In[66]:


perm_imp_xgb = PermutationImportance(xg, random_state=42,scoring='accuracy').fit(X_test, y_test)
eli5.show_weights(perm_imp_xgb, feature_names = data.iloc[:,:-1].columns.tolist(),top=50)


# In[93]:


data.iloc[:,:-1].columns


# In[67]:


#from sklearn.model_selection import train_test_split
#Input_train, Input_test, Target_train, Target_test = train_test_split(InputScaled, y, test_size = 0.30, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[68]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=13, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


# In[76]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=120, verbose=1)


# In[77]:


model.summary()

score = model.evaluate(X_test, y_test, verbose=0)

print('Keras Model Accuracy = ',score[1])


# In[78]:


y_pred = model.predict(X_test)


# In[79]:


y_pred_train = model.predict(X_train)


# In[80]:


y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0
y_pred_train[y_pred_train > 0.5] = 1
y_pred_train[y_pred_train < 0.5] = 1


# In[95]:


y_test


# In[81]:


cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(y_pred_train, y_train)


# In[87]:


print('Accuracy for test set for Keras = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

#Accuracy for training set for Logistic Regression = 0.8471074380165289
#Accuracy for test set for Logistic Regression = 0.7868852459016393

#Accuracy for training set for Random Forest = 0.987603305785124
#Accuracy for test set for Random Forest = 0.7540983606557377

#Accuracy for training set for XGBoost = 1.0
#Accuracy for test set for XGBoost = 0.7540983606557377

#Accuracy for training set for Keras Deep learning = 0.8554374787864554
#Accuracy for test set for Keras = 0.8032786885245902

#Feature Importance - Permutation Importance

#Logistic Regression
#Positive features = ['thal','ca','exang','cp','thalach','oldpeak']
#Negative features = ['restecg','chol','trestbps']
#Neutral features = ['sex','fbs','slope','age']

#Random Forest
#Positive = ['thal','ca','cp','sex']
#Negative = ['fbs','thalach','exang','oldpeak','restecg','chol','trestbps','age','slope']

#XgBoost
#Positive = ['ca','thal','cp','exang','sex']
#Neutral = ['thalach','chol','trestbps','age']
#Negative = ['oldpeak','restecg','fbs','slope']
# In[101]:


sizes = [len(data[data['thal'] == 3.0]), len(data[data['thal']==6.0]), len(data[data['thal']==7.0])]
labels = ['Normal', 'Fixed Defect', 'Reversible defect']
plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)
plt.show()


# In[102]:


f, axes = plt.subplots(1,2,figsize=(15,5))
absence = data.loc[data['num']==0,'thal']
presence = data.loc[data['num']==1,'thal']
sns.countplot(absence, data=data,ax=axes[0],order=[3.0, 6.0, 7.0]).set_title('Absence of Heart Disease')
sns.countplot(presence, data=data,ax=axes[1],order=[3.0, 6.0, 7.0]).set_title('Presence of Heart Disease')
plt.show()

