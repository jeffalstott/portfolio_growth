
# coding: utf-8

# In[4]:

import urllib.request

data_directory = '../data/'


# In[5]:

microsoft_url_base = 'https://academicgraph.blob.core.windows.net/graph-2016-02-05/'
microsoft_file_directory = 'Microsoft_Academic_Graph/'
files = ['PaperAuthorAffiliations.zip',
         ''
         ]
https://academicgraph.blob.core.windows.net/graph-2016-02-05/PaperReferences.zip
for file in files:
    urllib.request.urlretrieve(microsoft_url_base+file, data_directory+microsoft_file_directory+file)

