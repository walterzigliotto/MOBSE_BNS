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



def count_events_single_chunk(dataf, IDs_list, fMT, metallicity, chunk, list_values):

    #counter for yes CE phase
    counter = 0
    #counter no CE phase
    not_counter = 0
    
    #ID for which I have two CE phases, which cannot be
    #they are problematic ID
    prob_ID = []    

    #iterate over all possible IDs
    for single_event_ID in IDs_list:
        
        #here I retrieve a subset with only the systems I am interested in 
        single_system = dataf[dataf.iloc[:,0] == single_event_ID]
        n_rows = len(single_system)

        bool_check           = False
        CE_intermediate_flag = False
        CE_final_flag        = False
        
        
        #check if there is a CE label, if not CE has not happened and then continue with next ID
        bool_check = single_system.loc[:,'label[33]'].all() != 'COMENV'
        if bool_check == False: continue
        
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
#                     print("Intermediate CE!")
                
                #check for final CE phases
                if ((bool_both_NS) & (bool_single_star_past) & (was1_NS|was2_NS)):
                    CE_final_flag = True
                    counter_CEs += 1
#                     print("Final CE!")
        
    #if they have more than 2 CEs... let us save them!
        if (counter_CEs > 2):  prob_ID.append([single_event_ID, fMT, Z, chunk])
            
        if ((CE_intermediate_flag) | (CE_final_flag)):    
            counter += 1
            
        if ( (CE_intermediate_flag) & (CE_final_flag) != True):
            not_counter += 1
        
    list_values.append([counter, not_counter, prob_ID])
    return list_values



def compute_data(list_values, results, Z, fMT_to_csv, file_debug):
    
    #sum over all columns
    totals     =  np.sum(list_values, axis = 0)
    
    tot_CE        = totals[0] 
    tot_NOT_CE    = totals[1]
    tot_events    = totals[0] + totals[1]
        
    percents = {'CE_phase': tot_CE/tot_events, 'NOT_CE_phase' : tot_NOT_CE/tot_events ,"tot_events": tot_events , 'Z' : Z, 'fMT' : fMT_to_csv  }
    
    results = results.append(percents, ignore_index = True)
    
    for event_problem in totals[2]:
            file_debug = file_debug.append({'ID' : event_problem[0], 'fMT': event_problem[1], 'Z': event_problem[2], 'chunk': event_problem[3]}, ignore_index = True)

    return results, file_debug



def debug_dataframe(debug_df, fMT ):
    
    output_dataframe = pd.DataFrame()
    
    ID_list       = debug_df.iloc[:,0].unique()
    simulations   = debug_df.iloc[:,1].unique()
    metallicities = debug_df.iloc[:,2].unique()
    n_chunk       = debug_df.iloc[:,3].unique()
    
    
    for num_sim, fmt_csv in zip (fMT, simulations):
        for Z in metallicities:
            for chunk in n_chunk:
                dataset = import_single_chunk(num_sim, Z, chunk)
                for ID in ID_list:
                    filtered = dataset[dataset.iloc[:,0] == ID ]
                    filtered.loc[:,'Z']    = Z
                    filtered.loc[:,'fMT']  = fmt_csv          
                    output_dataframe = output_dataframe.append(filtered, ignore_index = True)
                    
    return output_dataframe
        
            
fMT = ['01','02','03','04','05','07','1'] 
metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunks = ['0','1','2','3','4']
fMT_csv   = ['0.1','0.2','0.3','0.4','0.5','0.7','1']

results   = pd.DataFrame(columns = ['CE_phase', 'NOT_CE_phase', "tot_events" , 'Z', 'fMT'] )    
debug_csv = pd.DataFrame(columns = ['ID', 'fMT', 'Z', 'chunk'])


for sim_num, fMT_to_csv in zip(fMT, fMT_csv):
    
    for Z in metallicities:
        
        start_time = time.time()
        list_temp = []
        
        for chunk in chunks:
            start_chunk = time.time()
            print("Starting for chunk", chunk, "of fMT", sim_num , "Z", Z)
            ALL_BNS_list = import_IDs_from_COB(sim_num, Z, chunk)
            #import data reduced
            data, ID_list = import_chunk_reduced(ALL_BNS_list, sim_num, Z, chunk)
            #count event for single chunk
            count_events_single_chunk(data, ID_list, fMT_to_csv, Z, chunk, list_temp)
            end_chunk = time.time()
            print("Finished for chunk", chunk, "of fMT", sim_num , "Z", Z, "with time:", end_chunk - start_chunk )
            
        results, debug_csv = compute_data(list_temp, results, Z, fMT_to_csv, debug_csv)
        
        final_time = time.time()
        print("Finished for fMT", sim_num , "Z", Z, "    with time:", final_time - start_time, "\n" )

results.to_csv('BNS/assignment02/Count_CE_def.csv', index=False)
debug_csv.to_csv('BNS/assignment02/debug.csv', index=False)

error_IDs = debug_dataframe(debug_csv, fMT)
error_IDs.to_csv('BNS/assignment02/deep_debug.csv', index=False)











