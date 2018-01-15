
# coding: utf-8

# In[ ]:

import urllib.request
from zipfile import ZipFile


# In[6]:

data_directory = '../data/'


# Research Article Data
# ===
# (from https://academicgraph.blob.core.windows.net/graph-2016-02-05/index.html)

# In[16]:

url_base = 'https://academicgraph.blob.core.windows.net/graph-2016-02-05/'

file_directory = 'Microsoft_Academic_Graph/'

files = ['FieldsOfStudy.zip',
         'FieldOfStudyHierarchy.zip',
         'Journals.zip',
         'Affiliations.zip',
         'PaperAuthorAffiliations.zip',
         'PaperReferences.zip',
         'Papers.zip',
         'PaperKeywords.zip'
         ]

from time import time
for file in files:
    print(file)
    t = time()
    urllib.request.urlretrieve(url_base+file, 
                               data_directory+file_directory+file)
    ZipFile(data_directory+file_directory+file).extractall(data_directory+file_directory)
    print(time()-t)
    


# In[17]:

url_base = 'http://www.patentsview.org/data/20160518/'

file_directory = 'PatentsView/'

files = ['cpc_current.zip',
         'cpc_group.zip',
         'cpc_subgroup.zip',
         'cpc_subsection.zip',
         'patent.zip',
         'uspatentcitation.zip',
         'inventor.zip',
         'patent_inventor.zip'
         ]

for file in files:
    print(file)
    t = time()
    urllib.request.urlretrieve(url_base+file, data_directory+file_directory+file)
    ZipFile(data_directory+file_directory+file).extractall(data_directory+file_directory)
    print(time()-t)

