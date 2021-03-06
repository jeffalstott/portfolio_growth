
# coding: utf-8

# In[1]:

import pandas as pd
get_ipython().magic('pylab inline')


# In[ ]:

df = df[['patent_id', 'group_id']]
df.rename(columns={'patent_id': 'Patent',
                   'group_id': 'Class_CPC4'},
          inplace=True)


# In[25]:

class_lookup = pd.read_csv(data_directory+'cpc_group.tsv', sep='\t')


# In[1]:

df['Class_CPC4'] = class_lookup.set_index('id').ix[df['Class_CPC4']]


# In[ ]:

df['Year'] = patent_years.ix[df['Patent']]


# In[ ]:

store = pd.HDFStore(data_directory+'classifications_organized.h5', mode='a', table=True)
store.put('/CPC4_class_lookup', class_lookup, 'table', append=False)
store.put('/patent_classes_CPC4', df, 'table', append=False)
store.close()


# In[ ]:

class_lookup = pd.DataFrame


# In[ ]:

IPC4_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'IPC4_class_lookup')

patent_classes_IPC4['Class_IPC4'] = IPC4_class_lookup.ix[patent_classes_IPC4['Class_IPC4']].values

patent_classes_IPC4['Class_IPC4'].dropna(inplace=True)


# Import Years
# ===

# In[2]:

# data_directory = '../data/'


# In[57]:

df = pd.read_csv(data_directory+'pid_issdate_ipc.csv',
                          index_col=0)


# In[58]:

df['Year'] = df.ISSDATE.map(lambda x: int(x[-4:]))
df.drop(['ISSDATE', 'IPC3'], axis=1, inplace=True)
patent_years = df


# In[59]:

patent_years = patent_years[patent_years['Year']<=2010]


# Import IPC classes
# ===

# In[64]:

patent_classes_IPC = pd.read_csv(data_directory+'patn_multi_ipc3_1976_2015_no_clean.csv')
patent_classes_IPC.rename(columns={'IPC3': "Class_IPC"},
                  inplace=True)

# patent_classes_IPC.ix[:,'Class_IPC'] = patent_classes_IPC['Class_IPC'].map(lambda x: x if type(x)==int else int(x) if x.isdigit() else nan)
# patent_classes_IPC.dropna(inplace=True)


# In[65]:

patent_classes_IPC = patent_years.merge(patent_classes_IPC[['PID', 'Class_IPC']],right_on='PID',left_index=True).set_index('PID')
patent_classes_IPC = patent_classes_IPC.reset_index().drop_duplicates().set_index('PID')


# In[66]:

# IPC_classes = sort(patent_classes_IPC['Class_IPC'].unique())
# IPC_class_lookup = pd.Series(index=IPC_classes,
#                       data=arange(len(IPC_classes)))
IPC_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'IPC_class_lookup')
patent_classes_IPC['Class_IPC'] = IPC_class_lookup.ix[patent_classes_IPC['Class_IPC']].values

patent_classes_IPC['Class_IPC'].dropna(inplace=True)


# Import IPC4 classes
# ===

# In[67]:

patent_classes_IPC4 = pd.read_csv(data_directory+'patn_multi_ipc4_1976_2015_no_clean.csv')
patent_classes_IPC4.rename(columns={'IPC4': "Class_IPC4"},
                  inplace=True)

# patent_classes_IPC.ix[:,'Class_IPC'] = patent_classes_IPC['Class_IPC'].map(lambda x: x if type(x)==int else int(x) if x.isdigit() else nan)
# patent_classes_IPC.dropna(inplace=True)


# In[68]:

patent_classes_IPC4 = patent_years.merge(patent_classes_IPC4[['PID', 'Class_IPC4']],right_on='PID',left_index=True).set_index('PID')
patent_classes_IPC4 = patent_classes_IPC4.reset_index().drop_duplicates().set_index('PID')


# In[69]:

# IPC4_classes = sort(patent_classes_IPC4['Class_IPC4'].unique())
# IPC4_class_lookup = pd.Series(index=IPC4_classes,
#                       data=arange(len(IPC4_classes)))
IPC4_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'IPC4_class_lookup')

patent_classes_IPC4['Class_IPC4'] = IPC4_class_lookup.ix[patent_classes_IPC4['Class_IPC4']].values

patent_classes_IPC4['Class_IPC4'].dropna(inplace=True)


# Import USPC classes
# ====

# In[70]:

patent_classes_USPC = pd.read_csv(data_directory+'PATENT_US_CLASS_SUBCLASSES_1975_2011.csv',
                               header=None,
                               names=['PID', 'Class_USPC', 'Subclass_USPC'])

patent_classes_USPC.ix[:,'Class_USPC'] = patent_classes_USPC['Class_USPC'].map(lambda x: x if type(x)==int else int(x) if x.isdigit() else nan)
patent_classes_USPC.dropna(inplace=True)


# In[71]:

patent_classes_USPC = patent_years.merge(patent_classes_USPC[['PID', 'Class_USPC']],right_on='PID',left_index=True).set_index('PID')
patent_classes_USPC = patent_classes_USPC.reset_index().drop_duplicates().set_index('PID')


# In[72]:

# USPC_classes = sort(patent_classes_USPC['Class_USPC'].unique())
# USPC_class_lookup = pd.Series(index=USPC_classes,
#                       data=arange(len(USPC_classes)))

USPC_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'USPC_class_lookup')
patent_classes_USPC['Class_USPC'] = USPC_class_lookup.ix[patent_classes_USPC['Class_USPC']].values


# Write Data
# ===

# In[73]:

store = pd.HDFStore(data_directory+'classifications_organized.h5', mode='a', table=True)


# In[74]:

store.put('/IPC_class_lookup', IPC_class_lookup, 'table', append=False)
store.put('/patent_classes_IPC', patent_classes_IPC, 'table', append=False)


# In[75]:

store.put('/IPC4_class_lookup', IPC4_class_lookup, 'table', append=False)
store.put('/patent_classes_IPC4', patent_classes_IPC4, 'table', append=False)


# In[76]:

store.put('/USPC_class_lookup', USPC_class_lookup, 'table', append=False)
store.put('/patent_classes_USPC', patent_classes_USPC, 'table', append=False)


# In[77]:

store.close()

