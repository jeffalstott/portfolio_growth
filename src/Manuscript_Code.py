
# coding: utf-8

# In[6]:

import pandas as pd
# %pylab inline
from pylab import *


# In[8]:

data_directory = '../data/'


# In[9]:

import os
def create_directory_if_not_existing(f):
    try:
        os.makedirs(f)
    except OSError:
        pass


# Organize data for citations, co-classifications and occurrences
# ===

# In[9]:

# print("Organizing Classifications")
# %run -i Organize_Classifications.py
# print("Organizing Citations")
# %run -i Organize_Citations.py
# print("Organizing Occurrences")
# %run -i Organize_Occurrences.py


# Define parameters
# ===

# Define classes and entities to analyze
# ---

# In[1]:

class_systems = ['CPC4']
occurrence_entities = {#'Firm': ('occurrences_organized.h5', 'entity_classes_Firm'),
                       'Inventor': ('occurrences_organized.h5', 'entity_classes_Inventor'),
                       'PID': ('classifications_organized.h5', 'patent_classes'),
                       }
entity_types = list(occurrence_entities.keys())


# Define what years to calculate networks for
# ---

# In[2]:

target_years = 'all'


# Define number of years of history networks should include
# ---

# In[3]:

all_n_years = ['all', 1, 5]

def create_n_years_label(n_years):
    if n_years is None or n_years=='all' or n_years=='cumulative':
        n_years_label = ''
    else:
        n_years_label = '%i_years_'%n_years
    return n_years_label


# In[4]:

citation_metrics = ['Class_Cites_Class_Count',
                    'Class_Cited_by_Class_Count']


# Calculate empirical networks
# ===

# In[5]:

create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/')
create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/citations/')
create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/cooccurrence/')


# In[15]:

# ### Create empirical networks
# randomized_control = False

# for class_system in class_systems:
#     for n_years in all_n_years:
#         print("Calculating for %s------"%class_system)
#         print("Calculating for %s years------"%str(n_years))
#         ### Calculate citation networks
#         %run -i Calculating_Citation_Networks.py
#         all_networks = networks

#         ### Calculate co-occurrence networks
#         preverse_years = True
#         for entity_column in entity_types:
#             target_years = 'all'
#             print(entity_column)
#             occurrence_data, entity_data = occurrence_entities[entity_column]
#             %run -i Calculating_CoOccurrence_Networks.py
#             all_networks.ix['Class_CoOccurrence_Count_%s'%entity_column] = networks

#         ind = ['Class_CoOccurrence_Count_%s'%entity for entity in entity_types]
#         store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/cooccurrence/class_relatedness_networks_cooccurrence.h5', 
#                         mode='a', table=True)
#         n_years_label = create_n_years_label(n_years)
#         store.put('/empirical_cooccurrence_%s%s'%(n_years_label,class_system), all_networks.ix[ind], 'table', append=False)
#         store.close()

#         #### Combine them both
#         store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/class_relatedness_networks.h5', 
#                             mode='a', table=True)
#         store.put('/empirical_'+n_years_label+class_system, all_networks, 'table', append=False)
#         store.close()


# Calculate randomized networks
# ====

# Make directories
# ---

# In[11]:

create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/citations/controls/')
create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/')


# Run randomizations
# ---
# (Currently set up to use a cluster)

# In[12]:

first_rand_id = 0
n_randomizations = 1000
overwrite = False


# In[13]:

# python_location = '/home/jeffrey_alstott/anaconda3/bin/python'
# from os import path
# abs_path_data_directory = path.abspath(data_directory)+'/'


# create_directory_if_not_existing('jobfiles/')

# for class_system in class_systems:
#     for n_years in all_n_years:
#         ### Citations
#         create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/citations/controls/'+class_system)
#         basic_program = open('Calculating_Citation_Networks.py', 'r').read()
#         job_type = 'citations'
#         options="""class_system = %r
# target_years = %r
# n_years = %r
# data_directory = %r
# randomized_control = True
# citation_metrics = %r
#     """%(class_system, target_years, n_years, abs_path_data_directory, citation_metrics)

#         %run -i Calculating_Synthetic_Networks_Control_Commands

#         ### Co-occurrences
#         create_directory_if_not_existing(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/'+class_system)
#         basic_program = open('Calculating_CoOccurrence_Networks.py', 'r').read()
#         job_type = 'cooccurrence'
#         for entity in entity_types:
#             occurrence_data, entity_data = occurrence_entities[entity]
#             options = """class_system = %r
# target_years = %r
# n_years = %r
# data_directory = %r
# randomized_control = True
# preserve_years = True
# chain = False
# occurrence_data = %r
# entity_data = %r
# entity_column = %r
# print(occurrence_data)
# print(entity_data)
# print(entity_column)
#     """%(class_system, target_years, n_years, abs_path_data_directory, occurrence_data, entity_data, entity)

#             %run -i Calculating_Synthetic_Networks_Control_Commands


# Integrate randomized data and calculate Z-scores
# ---
# Note: Any classes that have no data (i.e. no patents within that class) will create z-scores of 'nan', which will be dropped when saved to the HDF5 file. Therefore, the z-scores data will simply not includes these classes.

# In[17]:

# n_controls = n_randomizations

# output_citations = 'class_relatedness_networks_citations'
# # output_citations = False
# output_cooccurrence = 'class_relatedness_networks_cooccurrence'
# # output_cooccurrence = False
# combine_outputs = True


# for class_system in class_systems:
#     print(class_system)
#     for n_years in all_n_years:
#         print(n_years)
#         n_years_label = create_n_years_label(n_years)
#         cooccurrence_base_file_name = 'synthetic_control_cooccurrence_'+n_years_label+'%s_preserve_years_%s'

#         %run -i Calculating_Synthetic_Networks_Integrate_Runs.py


# Delete individual runs of randomized data
# ---

# In[10]:

# from shutil import rmtree

# for class_system in class_systems:
#     rmtree(data_directory+'Class_Relatedness_Networks/citations/controls/'+class_system)
#     rmtree(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/'+class_system)  


# Regress out popularity from relatedness measures
# ---
# First create popularity-by-year networks for all class systems and n_years

# In[16]:

# %run -i Calculating_Popularity_Networks.py


# In[17]:

# %run -i Regressing_Popularity_Out_of_Z_Scores.py


# Create inventor movement data
# ----

# Precalculate some data that all runs will rely on and which takes a long time to calculate

# In[18]:

# for class_system in class_systems:
#     %run -i Precalculating_Supporting_Data_for_Agent_Movement_Calculations.py


# In[ ]:

# python_location = '/home/jeffrey_alstott/anaconda3/bin/python'
# from os import path
# abs_path_data_directory = path.abspath(data_directory)+'/'
# create_directory_if_not_existing('jobfiles/')
# create_directory_if_not_existing(data_directory+'Agent_Movement/')

# for class_system in ['IPC4']:#class_systems:

#     store = pd.HDFStore(data_directory+'organized_patent_data.h5')
#     agents_lookup_movers = store['agents_lookup_movers_%s'%class_system]
#     store.close()

#     n_agents = agents_lookup_movers.shape[0]


#     sample_length = 1000
#     all_samples_end = n_agents+sample_length
#     all_samples_start = 0
#     overwrite = True
#     %run -i Calculating_Inventor_Movement_Data_Control_Commands.py


# Integrate the data on entries
# ===

# In[ ]:

# for class_system in class_systems:
#     %run -i Movement_Data_Integrate_Entries.py


# Precalculate data for descriptive statistics
# ===

# In[1]:

# for class_system in class_systems:
#     %run -i Precalculating_Data_for_Descriptive_Statistics.py


# Make figures of entries data
# ===

# In[ ]:

# for class_system in class_systems:
#     %run -i Manuscript_Figures.py


# Integrate the data on entries and non-entries
# ===

# In[ ]:

# label_columns = ['Agent_Class_Number', 
#                  'Agent_ID', 
#                  'Entered', 
#                  'Class_ID', 
#                  'Application_Year',] 
# #                  'Issued_Year']

# agent_properties = ['Agent_Number_of_Classes_All_Time',
#                     'Agent_Number_of_Patents_All_Time', 
#                     'Agent_Patent_Number',
#                     'Class_Diversity_Entropy',]
#                     #'Class_Diversity_Herfindahl_Index']

# target_columns = ['Class_Cites_Class_Count_Percentile', #Network
#                   'Class_Agent_Count_Previous_Year_Percentile', #Popularity
#                   'Agent_Previous_Citations_to_Class', #Personal history: Agent's own citations to the class*
#                   'CoAgent_Count_in_Class', #Personal history: Co-Agents previously in class
#                  ]+label_columns+agent_properties

# for class_system in class_systems:
#     %run -i Movement_Data_Integrate_Entries_and_NonEntries.py

