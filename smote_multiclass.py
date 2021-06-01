#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import multiprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.neural_network
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

from prettytable import PrettyTable

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


# In[45]:


df = pd.read_csv("data_traffic.txt",sep=",",names=["duration","protocoltype","service","flag","srcbytes","dstbytes","land", "wrongfragment","urgent","hot","numfailedlogins","loggedin", "numcompromised","rootshell","suattempted","numroot","numfilecreations", "numshells","numaccessfiles","numoutboundcmds","ishostlogin",
"isguestlogin","count","srvcount","serrorrate", "srvserrorrate",
"rerrorrate","srvrerrorrate","samesrvrate", "diffsrvrate", "srvdiffhostrate","dsthostcount","dsthostsrvcount","dsthostsamesrvrate", "dsthostdiffsrvrate","dsthostsamesrcportrate",
"dsthostsrvdiffhostrate","dsthostserrorrate","dsthostsrvserrorrate",
"dsthostrerrorrate","dsthostsrvrerrorrate","attack", "lastflag"])
df.head()


# In[46]:


df.drop(['land','urgent','numfailedlogins','numoutboundcmds'],axis=1,inplace=True)


# In[47]:


categories_col = [n for n in df.columns if df[n].dtypes == 'object']
for column in categories_col:
    print(column, '\n')
    print(df[column].value_counts())
    print("------------------------------------------------------------------")


# In[48]:


df['attack']=np.where(df['attack'] =='back', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='land', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='neptune', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='pod', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='smurf', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='teardrop', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='apache2', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='udpstorm', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='processtable', 'dos', df['attack'])
df['attack']=np.where(df['attack'] =='worm', 'dos', df['attack'])


# In[49]:


df['attack']=np.where(df['attack'] =='satan', 'probe', df['attack'])
df['attack']=np.where(df['attack'] =='ipsweep', 'probe', df['attack'])
df['attack']=np.where(df['attack'] =='nmap', 'probe', df['attack'])
df['attack']=np.where(df['attack'] =='portsweep', 'probe', df['attack'])
df['attack']=np.where(df['attack'] =='mscan', 'probe', df['attack'])
df['attack']=np.where(df['attack'] =='saint', 'probe', df['attack'])


# In[50]:


df['attack']=np.where(df['attack'] =='guess_passwd', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='ftp_write', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='imap', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='phf', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='multihop', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='warezmaster', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='warezclient', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='spy', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='xlock', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='xsnoop', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='snmpguess', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='snmpgetattack', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='httptunnel', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='sendmail', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='named', 'r2l', df['attack'])
df['attack']=np.where(df['attack'] =='mailbomb', 'r2l', df['attack'])


# In[51]:


df['attack']=np.where(df['attack'] =='buffer_overflow', 'u2r', df['attack'])
df['attack']=np.where(df['attack'] =='loadmodule', 'u2r', df['attack'])
df['attack']=np.where(df['attack'] =='rootkit', 'u2r', df['attack'])
df['attack']=np.where(df['attack'] =='perl', 'u2r', df['attack'])
df['attack']=np.where(df['attack'] =='sqlattack', 'u2r', df['attack'])
df['attack']=np.where(df['attack'] =='xterm', 'u2r', df['attack'])
df['attack']=np.where(df['attack'] =='ps', 'u2r', df['attack'])


# Классификация по 4 группам

# In[62]:


df['attack'].value_counts()


# In[52]:


sns.countplot(df['attack'])


# In[53]:


encor=LabelEncoder()
df['protocoltype']=encor.fit_transform(df['protocoltype'])
df['service']=encor.fit_transform(df['service'])
df['flag']=encor.fit_transform(df['flag'])


# In[54]:


X=df.drop(['attack'],axis=1)
y=df['attack']


# In[55]:


scaler = StandardScaler()
scaler.fit(X)
X_transformed = scaler.transform(X)


# SMOTE сбалансируем данные

# In[56]:


from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_resample(X, y)


# In[57]:


y_resampled.value_counts()


# In[58]:


X_re_train,X_re_test,y_re_train,y_re_test = train_test_split(X_resampled,y_resampled, test_size = 0.3 , random_state = 0)


# In[65]:


def evaluate_model(model, modelname):
    start_time = datetime.now()
    model.fit(X_re_train, y_re_train)
    y_re_pred = model.predict(X_re_test)
    time = datetime.now() - start_time
    print(modelname)
    print("Time execution: ", time)
    #print(confusion_matrix(y_re_test,y_re_pred))
    plot_confusion_matrix(model, X_re_test, y_re_test,cmap=plt.cm.Blues)
    print(classification_report(y_re_test,y_re_pred))


# SVM

# In[70]:


svc = svm.LinearSVC(random_state=0, loss='squared_hinge', penalty='l1', dual=False,class_weight='balanced')


# In[71]:


evaluate_model(svc, "SVM")


# RandomForest

# In[72]:


rf = RandomForestClassifier()


# In[73]:


evaluate_model(rf, "RandomForest")


# LogisticRegression

# In[74]:


lr = LogisticRegression(solver='liblinear', max_iter=100)


# In[75]:


evaluate_model(lr, "LogisticRegression")


# AdaBoost

# In[76]:


ab = AdaBoostClassifier()


# In[77]:


evaluate_model(ab, "AdaBoost")


# neural_network

# In[78]:


neural = sklearn.neural_network.MLPClassifier()


# In[79]:


evaluate_model(neural, "MLPClassifier")


# KNN

# In[80]:


#Не рассматриваем, так как у алгоритм большое время обучения 
#knn = KNeighborsClassifier()
#evaluate_model(knn, "KNN")


# Naive Bayes

# In[81]:


nbc = GaussianNB()
evaluate_model(nbc, "Naive Bayes")


# In[ ]:




