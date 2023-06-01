#!/usr/bin/env python
# coding: utf-8

# # Task1:- Iris Flower Classification (LetsGrowMore)

# ## By :- Rupali Rakhunde

# In[265]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[266]:


df = pd.read_csv('iris.csv')


# In[267]:


df.head()


# In[268]:


df.sample(5)


# In[269]:


df.shape


# In[270]:


df.isnull().sum()


# In[271]:


df.info()


# In[272]:


df.describe()


# In[273]:


#correlation between columns

df.corr()


# In[274]:


n = len(df[df['Species'] == 'setosa'])
print('The number of setosa species is ',n)


# In[275]:


n1 = len(df[df['Species'] == 'virginica'])
print('The number of virginica species is ',n1)


# In[276]:


n2 = len(df[df['Species'] == 'versicolor'])
print('The number of versicolor species is ',n2)


# In[277]:


sns.countplot(df['Species'])
df['Species'].value_counts()


# In[307]:


plt.hist(df['Species'])


# In[279]:


plt.scatter(df['Sepal.Length'], df['Sepal.Width'])


# In[280]:


plt.scatter(df['Petal.Length'], df['Petal.Width'])


# In[281]:


plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df['Sepal.Length'])
plt.title('SP')

plt.subplot(121)
sns.distplot(df['Sepal.Width'])
plt.title('SW')

plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df['Petal.Length'])
plt.title('PL')

plt.subplot(121)
sns.distplot(df['Petal.Width'])
plt.title('PW')

#plt.show()
plt.show()


# In[282]:


sns.scatterplot(x='Sepal.Length',y='Petal.Length',data=df,hue='Species')
plt.show()


# In[283]:


plt.figure(figsize=(12,8))

sns.barplot(df["Sepal.Length"],df["Sepal.Width"])


# In[284]:


plt.figure(figsize=(12,8))
x =df["Petal.Length"]
y =df['Petal.Width']

sns.barplot(x,y)


# In[285]:


#detecting  for outliars
cols = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']

plt.figure(figsize=(15,10))
plt.subplot(221)
sns.boxplot(df[cols[0]])
plt.subplot(222)
sns.boxplot(df[cols[1]])
plt.subplot(223)
sns.boxplot(df[cols[2]])
plt.subplot(224)
sns.boxplot(df[cols[3]])
plt.show()


# In[286]:


sns.boxplot(x='Species',y='Petal.Length',palette='husl',data=df)
sns.boxplot(x='Species',y='Petal.Width',palette='husl',data=df)


# In[287]:


sns.boxplot(x='Species',y='Sepal.Length',palette='husl',data=df)
sns.boxplot(x='Species',y='Sepal.Width',palette='husl',data=df)


# In[288]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='Petal.Length',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='Petal.Width',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='Sepal.Length',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='Sepal.Width',data=df)


# In[289]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.swarmplot(x='Species',y='Petal.Length',data=df, color='r' , alpha=.8)
plt.subplot(2,2,2)
sns.swarmplot(x='Species',y='Petal.Width',data=df, color='b' , alpha=.8)
plt.subplot(2,2,3)
sns.swarmplot(x='Species',y='Sepal.Length',data=df, color='g' , alpha=.8)
plt.subplot(2,2,4)
sns.swarmplot(x='Species',y='Sepal.Width',data=df, color='y' , alpha=.8)


# In[290]:


sns.FacetGrid(df , hue='Species' , palette='husl' , height = 4).map(sns.kdeplot,'Petal.Length').add_legend()
sns.FacetGrid(df , hue='Species' , palette='husl' , height = 4).map(sns.kdeplot,'Petal.Width').add_legend()
sns.FacetGrid(df , hue='Species' , palette='husl' , height = 4).map(sns.kdeplot,'Sepal.Length').add_legend()
sns.FacetGrid(df , hue='Species' , palette='husl' , height = 4).map(sns.kdeplot,'Sepal.Width').add_legend()


# In[291]:


sns.pairplot(df,hue='Species');


# In[292]:


X = df['Sepal.Length'].values.reshape(-1,1)
print(X)


# In[293]:


Y = df['Sepal.Width'].values.reshape(-1,1)
print(Y)


# In[294]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# In[295]:


X = df.drop(['Species'],axis=1)
y = df['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=7)


# In[296]:


X_train.head()


# In[297]:


y_test.head()


# In[298]:


X_test.head()


# In[299]:


#by using logistic Regreesion

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred1 = lr.predict(X_test)

acc1 = (accuracy_score(y_test,y_pred1))
print((acc1)*100)


# In[300]:


print(confusion_matrix(y_test,y_pred1))


# In[301]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred1))


# In[302]:


#Using Decision Tree

dt = DecisionTreeClassifier(random_state=6)

dt.fit(X_train,y_train)

y_pred2 = dt.predict(X_test)

acc2 = (accuracy_score(y_test,y_pred2))
print((acc2)*100)


# In[303]:


#Using Support Vector Machines

svc = SVC()
svc.fit(X_train,y_train)

y_pred3 = svc.predict(X_test)

acc3 = (accuracy_score(y_test,y_pred3))
print((acc3)*100)


# In[304]:


#Using KNN Neighbors

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train,y_train)

y_pred4 = knn.predict(X_test)

acc4 = (accuracy_score(y_test,y_pred4))
print((acc4)*100)


# In[305]:


#Using Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred5 = gnb.predict(X_test)

acc5 = (accuracy_score(y_test,y_pred5))
print((acc5)*100)


# In[306]:


plt.figure(figsize=(12,6))
res=pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.86,0.86,0.83,0.93,0.86]})
sns.barplot(data=res,x=res['Model'],y=res['Score'])


# ## KNN perform better on iris dataset compared to others algorithm 

# In[ ]:




