#!/usr/bin/env python
# coding: utf-8

# In[2]:


data=pd.read_csv(r'E:\Krish naik\python code\DBscan\mall.csv',encoding='latin1')


# In[3]:


data.head()


# In[8]:


X=data.iloc[:,3:]


# In[9]:


X


# In[10]:


from sklearn.cluster import DBSCAN


# In[12]:


dbscan=DBSCAN(eps=3,min_samples=4)


# In[13]:


model=dbscan.fit(X)


# In[15]:


labels=model.labels_


# In[16]:


labels


# In[17]:


from sklearn import metrics


# In[18]:


#identifying the points which makes up our core points
sample_cores=np.zeros_like(labels,dtype=bool)

sample_cores[dbscan.core_sample_indices_]=True

#Calculating the number of clusters

n_clusters=len(set(labels))- (1 if -1 in labels else 0)



print(metrics.silhouette_score(X,labels))


# In[ ]:




