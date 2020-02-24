
# coding: utf-8

# In[1]:


from fastai.tabular import *


# In[2]:


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')


# In[25]:


df.head(7)


# In[4]:


deep_var = 'salary'


# In[7]:


cont_list ,cat_list = cont_cat_split(df=df ,max_card=20 ,dep_var=deep_var)
cont_list ,cat_list


# In[8]:


cat_names = ['workclass', 'education' ,'marital-status' ,'occupation' ,'relationship' ,'race']
cont_names= ['age' ,'fnlwgt' ,'education-num']


# In[9]:


procs = [FillMissing, Categorify, Normalize]


# In[11]:


test = TabularList.from_df(df.iloc[600:1100].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(df ,path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                            .split_by_idx(list(range(600,1100)))
                            .label_from_df(cols=deep_var)
                            .databunch())


# In[12]:


data.show_batch(rows=15)


# In[13]:


model = tabular_learner(data, layers=[300,100], metrics=accuracy)


# In[14]:


model.lr_find()


# In[15]:


model.recorder.plot(suggestion=True)


# In[16]:


lr = 1e-01
model.fit(5, lr)


# In[23]:


row = df.iloc[6]


# In[24]:


model.predict(row)

