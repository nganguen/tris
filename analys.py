#!/usr/bin/env python
# coding: utf-8

# In[65]:


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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

from prettytable import PrettyTable


# Загрузка тестовых данных

# In[2]:


df = pd.read_csv("data_traffic.txt",sep=",",names=["duration","protocoltype","service","flag","srcbytes","dstbytes","land", "wrongfragment","urgent","hot","numfailedlogins","loggedin", "numcompromised","rootshell","suattempted","numroot","numfilecreations", "numshells","numaccessfiles","numoutboundcmds","ishostlogin",
"isguestlogin","count","srvcount","serrorrate", "srvserrorrate",
"rerrorrate","srvrerrorrate","samesrvrate", "diffsrvrate", "srvdiffhostrate","dsthostcount","dsthostsrvcount","dsthostsamesrvrate", "dsthostdiffsrvrate","dsthostsamesrcportrate",
"dsthostsrvdiffhostrate","dsthostserrorrate","dsthostsrvserrorrate",
"dsthostrerrorrate","dsthostsrvrerrorrate","attack", "lastflag"])
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# 'land','urgent','numfailedlogins','numoutboundcmds' имеют в основном нулевые значения, то можеи удалить эти столбцы. Они не влияют на классификацию.

# In[5]:


df.drop(['land','urgent','numfailedlogins','numoutboundcmds'],axis=1,inplace=True)


# In[6]:


df.isna().sum()


# Количество столбцов с категориальными значениями

# In[7]:


df.select_dtypes(exclude=[np.number])


# Рассматриваем бинарную классификацию, то сделаем 2 категории: normal и attack. Закодируем эти стоблцы

# In[8]:


df['attack'].unique()


# In[9]:


df['attack'].loc[df['attack']!='normal']='attack'


# In[10]:


df['protocoltype'].unique()


# In[11]:


df['service'].unique()


# In[12]:


df['flag'].unique()


# In[13]:


encor=LabelEncoder()


# In[14]:


df['protocoltype']=encor.fit_transform(df['protocoltype'])
df['service']=encor.fit_transform(df['service'])
df['flag']=encor.fit_transform(df['flag'])


# Закодируем нормальный трафик normal - 0, аномалии - 1

# In[15]:


df['attack']=np.where(df['attack'] =='normal', '0', df['attack'])
df['attack']=np.where(df['attack'] =='attack', '1', df['attack'])


# In[16]:


df['attack'].unique()


# In[17]:


df['attack'].value_counts()


# По результату видим, что выборка достаточно сбалансирована. Пригодна для дальнейшего анализа.

# Построим матрицу корреляции 

# In[18]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr())


# Выделяем входные и выходные параметры

# In[19]:


X=df.drop(['attack'],axis=1)
y=df['attack']


# In[20]:


X.shape


# In[21]:


sns.countplot(df['attack'])


# In[22]:


scaler = StandardScaler()
scaler.fit(X)
X_transformed = scaler.transform(X)


# In[66]:


#функция для вычисления метрик: accurancy, precision, recall, f-mera
def class_report(model, X_test, y_test, y_pred):
    result = []
    start_time = datetime.now()
    result.append(model.score(X_test,y_test))
    time = datetime.now() - start_time
    result.append(precision_score(y_test, y_pred, average='binary', pos_label='1'))
    result.append(recall_score(y_test, y_pred, average='binary', pos_label='1'))
    result.append(f1_score(y_test, y_pred, average='binary', pos_label='1'))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result.append(tn)
    result.append(fp)
    result.append(fn)
    result.append(tp)
    result.append(time)
    return result


# Выделяем 70% под обучающей, 30% - тестовой

# In[55]:


X_train,X_test,y_train,y_test = train_test_split(X_transformed,y, test_size = 0.3 , random_state = 0)
y_test.value_counts()


# In[56]:


#для сохранения метрик моделей
metric_results_model = {}
#время выполнения прогнозирования
time_execution = {}


# Логистическая регрессия

# In[59]:


lr = LogisticRegression(solver='liblinear', max_iter=100)
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)


# Матрица ошибок

# In[60]:


print("confusion_matrix LogisticRegression")
confusion_matrix(y_test,y_pred_lr)


# In[61]:


print(classification_report(y_test,y_pred_lr))


# In[67]:


metric_results_model['LogisticRegression'] = class_report(lr, X_test, y_test, y_pred_lr)


# Рассмотрим метод опорных векторов

# In[68]:


svc = svm.LinearSVC(random_state=0, loss='squared_hinge', penalty='l1', dual=False,class_weight='balanced')
svc.fit(X_train, y_train)
y_pred_svm = svc.predict(X_test)


# In[69]:


print("confusion_matrix SVM")
confusion_matrix(y_test,y_pred_svm)


# In[70]:


print(classification_report(y_test,y_pred_svm))


# In[71]:


metric_results_model['SVM'] = class_report(svc, X_test, y_test, y_pred_svm)


# Рассмотрим случайный лес

# In[72]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[73]:


print("confusion_matrix RandomForest")
confusion_matrix(y_test,y_pred_rf)


# In[74]:


print(classification_report(y_test,y_pred_rf))


# In[75]:


metric_results_model['RandomForest'] = class_report(rf, X_test, y_test, y_pred_rf)


# AdaBoost

# In[76]:


ab = AdaBoostClassifier()
ab.fit(X_train, y_train)
y_pred_ab = ab.predict(X_test)


# In[77]:


print("confusion_matrix AdaBoost")
confusion_matrix(y_test,y_pred_ab)


# In[78]:


print(classification_report(y_test,y_pred_ab))


# In[79]:


metric_results_model['AdaBoost'] = class_report(ab, X_test, y_test, y_pred_ab)


# Метод knn

# In[80]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[81]:


print("confusion_matrix KNN")
confusion_matrix(y_test,y_pred_knn)


# In[82]:


print(classification_report(y_test,y_pred_knn))


# In[83]:


metric_results_model['KNN'] = class_report(knn, X_test, y_test, y_pred_knn)


# Наивный байесовский классификатор

# In[84]:


nbc = GaussianNB()
nbc.fit(X_train, y_train)
y_pred_nbc = nbc.predict(X_test)


# In[85]:


print("confusion_matrix Naive Bayes")
confusion_matrix(y_test,y_pred_nbc)


# In[86]:


print(classification_report(y_test,y_pred_nbc))


# In[87]:


metric_results_model['Naive Bayes'] = class_report(nbc, X_test, y_test, y_pred_nbc)


# In[90]:


result_table = PrettyTable()
result_table.field_names = ["Модель", "accurancy", "precision", "recall", "f1-score", "TN", "FP", "FN", "TP", "time"]
for modelname, metric in metric_results_model.items():
    result_table.add_row([modelname, '{:.3f}'.format(metric[0]), '{:.3f}'.format(metric[1]), '{:.3f}'.format(metric[2]), '{:.3f}'.format(metric[3]), metric[4], metric[5], metric[6], metric[7], metric[8]])
print(result_table)


# In[ ]:




