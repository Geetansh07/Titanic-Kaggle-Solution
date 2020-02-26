#!/usr/bin/env python
# coding: utf-8

# # WHAT MY DATA LOOKS LIKE

# In[1]:


#Let's see what the data look's like
import pandas as pd
df = pd.read_csv("G:\Lets do this\Machine Learning\Projects\Kaggle\Titanic\Train.csv")
df.head(10)


# # ANALYZING DATA

# In[2]:


import seaborn as sns
sns.countplot('Survived',data=df)


# In[3]:


sns.countplot('Survived',hue='Sex',data=df)


# # CLEANING THE DATA

# In[4]:


#there are many null value
df.isnull().sum()


# In[5]:


#so there are 177 null value in age and 687! null value in Cabin column
#Let's remove that data
df


# In[6]:


df.drop("Cabin",axis=1,inplace=True)


# In[7]:


df


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.isnull().sum()


# # NOW THE DATA IS CLEAN WITH NO NULL VALUES.....BUT THERE ARE MANY STRING VALUES WHICH CANNOT BE COMPUTED, SO WE WILL HAVE TO MAKE THEIR DUMMIES

# In[10]:


pd.get_dummies(df['Sex'])


# In[11]:


sex = pd.get_dummies(df['Sex'],drop_first=True)
sex.head()


# In[12]:


pclass = pd.get_dummies(df['Pclass'],drop_first=True)
pclass.head()


# In[13]:


df = pd.concat([df,sex,pclass],axis=1)
df.head()


# ### Now the data is clean and the categorical features have been changed into numerical values
# ### Now lets drop those columns which are of no use in predicting who survived

# In[14]:


df.drop(['Sex','Embarked','Name','Ticket','Parch','PassengerId','Pclass'],axis=1,inplace=True)


# In[15]:


df.head(10)


# ### LETS TRAIN THE BEAUTIFUL DATA

# In[16]:


X = df.drop('Survived',axis=1)
y = df['Survived']


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[18]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[19]:


model.fit(X_train,y_train)


# In[20]:


predictions = model.predict(X_test)


# #### LETS LOOK WHAT IS THE ACCURACY OF OUR MODEL

# In[21]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[22]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# # Importing our test set
# 

# In[100]:


df_test = pd.read_csv("G:\Lets do this\Machine Learning\Projects\Kaggle\Titanic\Test.csv")


# In[101]:


df_test.head()
df_test.isnull().sum()
df_test.info()


# In[102]:


df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean())
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].mean())
df_test.isnull().sum()


# In[103]:


df_test.drop(['Cabin','Embarked','Name','Ticket','Parch'],axis=1,inplace=True)
df_test.head()
df_test.info()


# In[104]:


sex1 = pd.get_dummies(df_test['Sex'],drop_first=True)
sex1.head()


# In[105]:


df_test.drop(['Sex'],axis=1,inplace=True)


# In[106]:


pclass1_1 = pd.get_dummies(df_test['Pclass'],drop_first=True)
pclass1_1.head()


# In[107]:


df_test = pd.concat([df_test,sex1,pclass1_1],axis=1)


# In[108]:


df_test.drop(['Pclass'],axis=1,inplace=True)
df_test.head()


# In[109]:


df_test.info()


# In[110]:


X = df.drop('Survived',axis=1)
y = df['Survived']
model_test = LogisticRegression()
model_test.fit(X,y)


# In[111]:


df_model_test = df_test.drop("PassengerId",axis=1).copy()
df_model_test.isnull().sum()
predictions2 = model_test.predict(df_model_test)


# In[112]:


submission = pd.DataFrame({
    "PassengerId":df_test["PassengerId"],
    "Survived": predictions2
})



# In[113]:


submission.head()

submission["PassengerId"] = submission["PassengerId"].astype(int)


# In[116]:


submission.isnull().sum()
submission.info()


# In[117]:


submission.to_csv('G:\Lets do this\Machine Learning\Projects\Kaggle\Titanic\submission.csv',index=False)


# In[118]:


submission = pd.read_csv('submission.csv')
submission.head(400)


# In[120]:


submission.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




