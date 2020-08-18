import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.lines import Line2D   
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

fMT = ['01','02','03','04','05','07','1']
fMT_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.7', '1']
fMT_csv   = ['0.1','0.2','0.3','0.4','0.5','0.7','1']

chunks = ['0', '1', '2', '3', '4']
metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']


#Creates a directory. equivalent to using mkdir -p on the command line
def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

    
def import_database_ID_nCEs():
    
    filename = 'BNS/assignment02/IDs_and_CEs_flags.csv'
    df = pd.read_csv(filename, header = 0, dtype = {'fMT': object, 'Z' : object, 'chunk': object})

    return df


def import_database(filename):
    
    path = 'BNS/assignment02/'
    df = pd.read_csv(path+filename, header = 0, dtype = {'fMT': object, 'Z' : object, 'chunk': object})

    return df


def import_single_chunk_givenIDs_evol_mergers(fMT, metallicity, chunk, dataset_info):
        
    filename = '../modB/simulations_fMT'+fMT+'/A5/'+metallicity+'/chunk'+chunk+'/evol_mergers.out'
    df = pd.read_csv(filename, delim_whitespace = True, header = 0)
    df_modified = df[df.columns[:-1]]
    df_modified.columns = df.columns[1:]
    
    IDs_flags_2A_2B = dataset_info[(dataset_info['2A_CE'] == True) & (dataset_info['2B_CE'] == True)].loc[:,'ID'].unique()
    
    #this function keeps all the rows with certain IDs
    #i.e. the ones in mergers file for the BNS systems
    df_subset_IDs = df_modified[df_modified.iloc[:,0].isin(IDs_flags_2A_2B)]
    
    #insert in the returning dataset the information about whether it has done CE
    #we insert it as a list so it ignores index of the dataframe
    count_CE = list(np.sum(dataset_info[dataset_info.iloc[:,0].isin(IDs_flags_2A_2B)].iloc[:,4:9].astype(bool), axis = 1))
        
    times_list = []
           
    for single_id, index in zip(IDs_flags_2A_2B, np.arange(len(count_CE))):

        if  count_CE[index] == 2:
                        
            df_subset_single_ID = df_subset_IDs[df_subset_IDs['ID[0]'] == single_id]
            time                = np.array(df_subset_single_ID[df_subset_single_ID['label[33]'] == 'COMENV'].loc[:,'t_step[1]'])
   
            times_list.append(time[1]-time[0])
                        
        elif count_CE[index] == 3:
                                    
            df_subset_single_ID = df_subset_IDs[df_subset_IDs['ID[0]'] == single_id]
            time                = np.array(df_subset_single_ID[df_subset_single_ID['label[33]'] == 'COMENV'].loc[:,'t_step[1]'])
            
            times_list.append(time[2]-time[1])
                         
        else : continue

    times_df = pd.DataFrame({'time' : times_list, 'N_CEs' : count_CE, 'fMT' : fMT, 'metallicity' : metallicity})
    
    return times_df




dataf_indexes = import_database_ID_nCEs()


dataf_times  = []

for sim_num, fMT_to_csv in zip(fMT, fMT_csv):
    list_temp_dataframesB_times = []
    
    for Z in metallicities:
        
        start_time = time.time()
        list_temp_dataframesA_times = []
        
        for chunk in chunks:
            
            start_chunk = time.time()
            
            print("Importing chunk", chunk, "of fMT", sim_num , "Z", Z)
            
            subindices = dataf_indexes[(dataf_indexes.loc[:,'chunk'] == chunk)&(dataf_indexes.loc[:,'Z'] == Z )&(dataf_indexes.loc[:,'fMT'] == sim_num)]
            
            data_temp_times = import_single_chunk_givenIDs_evol_mergers(sim_num, Z, chunk, subindices)
            
            end_chunk = time.time()
            
            print("Importing chunk", chunk, "of fMT", sim_num , "Z", Z, "with time:", end_chunk - start_chunk )
            list_temp_dataframesA_times.append(data_temp_times)

        dataf_givenZ_givenFMT_times = pd.concat(list_temp_dataframesA_times, ignore_index = True)  
        list_temp_dataframesB_times.append(dataf_givenZ_givenFMT_times)

    dataf_givenFMT_times = pd.concat( list_temp_dataframesB_times, ignore_index = True) 
    dataf_times.append(dataf_givenFMT_times)
    
dataf_times = pd.concat(dataf_times, ignore_index = True)


dataf_times.to_csv('BNS/assignment02/diff_times_CE.csv', index=False)



