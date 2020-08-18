import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.lines import Line2D   


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

            
def import_IDs_from_COB(fMT, metallicity, chunk):
    
    filename = '../modB/simulations_fMT'+fMT+'/A5/'+metallicity+'/chunk'+chunk+'/COB.out'
    df = pd.read_csv(filename, delim_whitespace = True, header = 0)
    df_modified = df[df.columns[:-1]]
    df_modified.columns = df.columns[1:]
    
    #check over COBs that are neutron stars
    df_NS = df_modified[(df_modified['k1form[7]'] == 13) & (df_modified['k2form[9]'] == 13)]
    BNS_systems_IDs = df_NS.loc[:,'ID[1]'].unique()
    
    return BNS_systems_IDs


def import_chunk_reduced(ALL_IDs, fMT, metallicity, chunk):
    
    filename = '../modB/simulations_fMT'+fMT+'/A5/'+metallicity+'/chunk'+chunk+'/evol_mergers.out'
    df = pd.read_csv(filename, delim_whitespace = True, header = 0)
    df_modified = df[df.columns[:-1]]
    df_modified.columns = df.columns[1:]
    
    #this function keeps all the rows with certain IDs
    #i.e. the ones in mergers file for the BNS systems
    df_subset_IDs = df_modified[df_modified.iloc[:,0].isin(ALL_IDs)]
    
    #systems that will remain are both Binary neutron stars 
    list_merging_BNS = df_subset_IDs.iloc[:,0].unique()
    
    return df_subset_IDs, list_merging_BNS


def check_common_envelope(single_event_ID, dataf):
    
    #here I retrieve a subset with only the systems I am interested in 
    single_system = dataf[dataf.iloc[:,0] == single_event_ID]
    n_rows = len(single_system)

    bool_check           = False
    CE_intermediate_flag = False
    CE_final_flag        = False
        
        
    #check if there is a CE label, if not CE has not happened
    bool_check = single_system.loc[:,'label[33]'].all() != 'COMENV'
    if bool_check == False: 
        return 0
        
    #now we need to analyze the CE labels
    counter_CEs = 0
        
    for row in range(n_rows):

        if single_system.iloc[row,33] == 'COMENV':
            #need to reset the flags at every iteration
            was1_NS = False
            was2_NS = False
                
                
            #this will be true for INTERMEDIATE_CE
            #check whether one of the two stars is at the moment a NS
            is1_NS = single_system.iloc[row,2] == 13
            is2_NS = single_system.iloc[row,16] == 13
            bool_single_NS = (is1_NS)^(is2_NS)
                
            #this will be true for FINAL_CE
            #check whether both are 
            bool_both_NS   = (is1_NS)&(is2_NS)
                
            #for both CE types, it must be true that they had to be a star in row BEFORE
            was1_star = single_system.iloc[row - 1,2] <= 12 
            was2_star = single_system.iloc[row - 1,16] <= 12 
            bool_single_star_past = (was1_star)^(was2_star)
                
            #for both CE types, it must be true that one of them used to be a NS also before 
            if is1_NS: was1_NS = single_system.iloc[row - 1,2] == 13
            if is2_NS: was2_NS = single_system.iloc[row - 1,16] == 13
                
            #check for intermediate CE phases
            if ((bool_single_NS) & (bool_single_star_past) & (was1_NS|was2_NS)):
                CE_intermediate_flag = True
                counter_CEs += 1
#               print("Intermediate CE!")
                
                #check for final CE phases
            if ((bool_both_NS) & (bool_single_star_past) & (was1_NS|was2_NS)):
                CE_final_flag = True
                counter_CEs += 1
#               print("Final CE!")
    
    #if more than 2, return the number itself and maybe we can take a look at it later...
    if (counter_CEs > 2): 
        return counter_CEs
    
    if ((CE_intermediate_flag) & (CE_final_flag)):    
        return 2
        
    if ((CE_intermediate_flag) | (CE_final_flag)):    
        return 1
        
    if ((CE_intermediate_flag) & (CE_final_flag) != True):
        return 0
            
    return counter_CEs
    


fMT = ['01','02','03','04','05','07','1'] 
metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunks = ['0','1','2','3','4']

columns_names = ['ID', 'fMT', 'Z', 'chunk', 'n_CE' ]
CE_general_csv = pd.DataFrame(columns = columns_names)

dataframe_listC = []

for sim_num in fMT:
    
    dataframes_listB =[]
    
    for Z in metallicities:
        
        start_time = time.time()
        
        dataframes_listA = []
        
        for chunk in chunks:
            start_chunk = time.time()
            print("Starting for chunk", chunk, "of fMT", sim_num , "Z", Z)
            ALL_BNS_list = import_IDs_from_COB(sim_num, Z, chunk)
            
            #import data reduced and the IDs that merge
            data, ID_list = import_chunk_reduced(ALL_BNS_list, sim_num, Z, chunk)
            
            CE_list = []
            
            for ID in ID_list:
                CE_list.append(check_common_envelope(ID, data))
 
            end_chunk = time.time()
            print("Finished for chunk", chunk, "of fMT", sim_num , "Z", Z, "with time:", end_chunk - start_chunk )
            
            temp_dataframeA = pd.DataFrame( {'ID' : ID_list,'fMT' : sim_num, 'Z': Z , 'chunk' : chunk, 'n_CE' : CE_list }  ,columns = columns_names)
            dataframes_listA.append(temp_dataframeA)
        
        dataframe_tempB = pd.concat(dataframes_listA, ignore_index = True)
        dataframes_listB.append(dataframe_tempB)

        final_time = time.time()
        print("Finished for fMT", sim_num , "Z", Z, "    with time:", final_time - start_time, "\n" )
    
    dataframe_tempC = pd.concat(dataframes_listB, ignore_index = True)
    dataframe_listC.append(dataframe_tempC)

CE_general_csv = pd.concat(dataframe_listC)
    
start_time = time.time()
print("Creating the final dataframe")
CE_general_csv.to_csv('BNS/assignment02/IDs_and_CEs.csv', index=False)
final_time = time.time()
print("Finished creating database with time:", final_time - start_time)
