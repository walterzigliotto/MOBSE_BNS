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


def check_common_envelope(single_event_ID, dataf, listCE1A,listCE1B,listCE1C, listCE2A, listCE2B):
    
    #here I retrieve a subset with only the systems I am interested in 
    single_system = dataf[dataf.iloc[:,0] == single_event_ID]
    n_rows = len(single_system)

    bool_check = False
    CE_1A_flag = False
    CE_1B_flag = False
    CE_1C_flag = False
    CE_2A_flag = False
    CE_2B_flag = False
        
    #check if there is a CE label, if not CE has not happened
    bool_check = single_system.loc[:,'label[33]'].all() != 'COMENV'
    if bool_check == False: 
        listCE1A.append(CE_1A_flag)
        listCE1B.append(CE_1B_flag)
        listCE1C.append(CE_1C_flag)
        listCE2A.append(CE_2A_flag)
        listCE2B.append(CE_2B_flag)        
        return 
                
    for row in range(n_rows):

        if single_system.iloc[row,33] == 'COMENV':
            
            #first type: both WERE stars in the row before
            was1_star = single_system.iloc[row - 1 ,2] <= 12 
            was2_star = single_system.iloc[row - 1,16] <= 12 
            bool_both_stars_past = (was1_star) & (was2_star)
            
            #check whether one of the object was a NS 
            #(true for 1B,2A)            
            was1_NS = single_system.iloc[row - 1 ,2] == 13
            was2_NS = single_system.iloc[row - 1,16] == 13           
            bool_single_NS_past = (was1_NS)^(was2_NS)
            
            #check whether only one was a STAR in the past (second type CE)
            bool_single_star_past = (was1_star)^(was2_star)
            
            #1A is when there are NO NS and both ARE stars
            is1_star = single_system.iloc[row,2] <= 12 
            is2_star = single_system.iloc[row,16] <= 12 
            bool_both_stars = (is1_star) & (is2_star)
            
            #check whether one of the two stars is NOW a NS 
            #(true for 1B,2A)
            is1_NS = single_system.iloc[row,2] == 13
            is2_NS = single_system.iloc[row,16] == 13
            bool_single_NS = (is1_NS)^(is2_NS)
            
            #check whether both are NS (true for 1C, 2B)
            bool_both_NS   = (is1_NS)&(is2_NS)
            
            #1A CE check
            #star + star -> star' + star'
            if ((bool_both_stars_past)&(bool_both_stars)):
                CE_1A_flag = True
            
            #1B CE check
            #star + star -> NS + star'
            if ((bool_both_stars_past)&(bool_single_NS)):
                CE_1B_flag = True
            
            #1C CE check
            #star + star -> NS + NS
            if ((bool_both_stars_past)&(bool_both_NS)):
                CE_1C_flag = True
            
            #2A CE check
            #NS + star   -> NS + star'
            if ((bool_single_NS_past)&(bool_single_star_past)&(bool_single_NS)):
                CE_2A_flag = True            
            
            #2B CE check
            #NS + star   -> NS + NS
            if ((bool_single_NS_past)&(bool_single_star_past)&(bool_both_NS)):
                CE_2B_flag = True
                
    listCE1A.append(CE_1A_flag)
    listCE1B.append(CE_1B_flag)
    listCE1C.append(CE_1C_flag)
    listCE2A.append(CE_2A_flag)
    listCE2B.append(CE_2B_flag)   
    
    return

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.lines import Line2D   


fMT = ['01','02','03','04','05','07','1'] 
#metallicities = ['0.016','0.02']
metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunks = ['0','1','2','3','4']

columns_names = ['ID', 'fMT', 'Z', 'chunk', '1A_CE','1B_CE','1C_CE','2A_CE','2B_CE' ]
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
            
            CE_1A  = []
            CE_1B  = []
            CE_1C  = []
            CE_2A  = []
            CE_2B  = []
            
            for ID in ID_list:
                check_common_envelope(ID, data, CE_1A, CE_1B, CE_1C, CE_2A, CE_2B)
 
            end_chunk = time.time()
            print("Finished for chunk", chunk, "of fMT", sim_num , "Z", Z, "with time:", end_chunk - start_chunk )
            
            temp_dataframeA = pd.DataFrame({'ID' : ID_list,'fMT' : sim_num, 'Z': Z , 'chunk' : chunk,
                                            '1A_CE' : CE_1A, '1B_CE' : CE_1B, '1C_CE': CE_1C,
                                            '2A_CE' : CE_2A, '2B_CE' : CE_2B },
                                           columns = columns_names)
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
CE_general_csv.to_csv('BNS/assignment02/IDs_and_CEs_flags.csv', index=False)
final_time = time.time()
print("Finished creating database with time:", final_time - start_time)
