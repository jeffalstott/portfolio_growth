
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sns
get_ipython().magic('pylab inline')


# In[2]:

data_directory = '../data/'


# In[3]:

df = pd.read_csv(data_directory+'cpc_current.tsv', sep='\t')


# In[4]:

df['subgroup_aggregated_id'] = df['subgroup_id'].map(lambda x: x.split('/')[0])


# In[5]:

q = df.groupby(['patent_id'])['group_id'].nunique()
z = q.value_counts().sort_index()
z.cumsum()/z.sum()


# In[6]:

r = df.groupby(['patent_id'])['subgroup_aggregated_id'].nunique()
z = r.value_counts().sort_index()
z.cumsum()/z.sum()


# In[7]:

r = df.groupby(['patent_id'])['subgroup_id'].nunique()
z = r.value_counts().sort_index()
z.cumsum()/z.sum()


# In[8]:

df_primary = df[df['category']=='primary']


# In[9]:

q = df_primary.groupby(['patent_id'])['group_id'].nunique()
z = q.value_counts().sort_index()
z.cumsum()/z.sum()


# In[10]:

r = df_primary.groupby(['patent_id'])['subgroup_aggregated_id'].nunique()
z = r.value_counts().sort_index()
z.cumsum()/z.sum()


# In[11]:

r = df_primary.groupby(['patent_id'])['subgroup_id'].nunique()
z = r.value_counts().sort_index()
z.cumsum()/z.sum()


# In[12]:

df.groupby(['patent_id'])['subgroup_aggregated_id'].nunique().mean()


# In[13]:

df_primary.groupby(['patent_id'])['subgroup_aggregated_id'].nunique().mean()


# In[14]:

import powerlaw
powerlaw.plot_cdf(df_primary['group_id'].value_counts().values)
yscale('linear')
xscale('log')
print(df_primary['group_id'].nunique())


# In[15]:

import powerlaw
z = df_primary['subgroup_aggregated_id'].value_counts()
powerlaw.plot_cdf(z.values)
yscale('linear')
xscale('log')
print(len(z))
mean(z>sqrt(len(z)))**2


# In[16]:

import powerlaw
powerlaw.plot_cdf(df_primary['subgroup_id'].value_counts().values)
yscale('linear')
xscale('log')


# In[112]:

patent_years = pd.read_csv(data_directory+'patent.tsv', sep='\t', usecols=['id', 'date'])


patent_years['id'] = pd.to_numeric(patent_years['id'], errors='coerce')

patent_years['date'] = pd.to_datetime(patent_years['date'], errors='coerce')
patent_years['year'] = pd.DatetimeIndex(patent_years['date']).year

patent_years.dropna(inplace=True)
patent_years.set_index('id', inplace=True)


# In[171]:

q = df['subgroup_aggregated_id'].unique()


# In[172]:

q = pd.SparseDataFrame(columns=q,index=q)


# In[ ]:

q.density


# In[97]:

def cooccurrence_counts(AB, A,B):
    import scipy.sparse
    cooccurrences = scipy.sparse.csr_matrix((ones_like(AB[A]),
                                                      (AB[A], 
                                                       AB[B])))

    present_cooccurrence = (cooccurrences.T * cooccurrences)
    
    all_cooccurrences = zeros((max(classes)+1, max(classes)+1))
    all_cooccurrences[:present_cooccurrence.shape[0], 
                          :present_cooccurrence.shape[1]] = present_cooccurrence
    
    return all_cooccurrences


# In[99]:

from scipy.sparse import csr_matrix


# Citations Counts
# ===

# In[23]:

citations = pd.read_csv(data_directory+'uspatentcitation.tsv', sep='\t')


# In[82]:

patent_forward_citations = citations['citation_id'].value_counts().reset_index()

patent_forward_citations.rename(columns={'index':'patent_id',
                                         'citation_id':'count'
                                         },
                                inplace=True
                               ) 

patent_forward_citations['patent_id'] = pd.to_numeric(patent_forward_citations['patent_id'],
                                                     errors='coerce')

patent_forward_citations.dropna(inplace=True)

patent_forward_citations.set_index('patent_id', inplace=True)


# In[95]:

from gc import collect
del(citations)
collect()

