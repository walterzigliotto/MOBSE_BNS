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

    
def import_database_ID_nCEs():
    
    filename = 'BNS/assignment02/IDs_and_CEs_flags.csv'
    df = pd.read_csv(filename, header = 0, dtype = {'fMT': object, 'Z' : object, 'chunk': object})

    return df


def plot_DT_fixedZ_CE_flag(dataframe, metallicity):
        
    output_dir = "BNS/assignment02/CE_fixedZ/CEflag"
    mkdir_p(output_dir)
    
    #make a color palette of 7 colors
    sns.set_palette('inferno', 7)
    
    #initialize space for plots:
    #1 row and 2 columns
    fig_times, axtimes    = plt.subplots(1,2,figsize = (30,12), sharey = True)    
    
    subdata = dataframe[dataframe['metallicity[21]'] == metallicity]
    
    #now set the log binning set on the min and the max 
    min_time, max_time = subdata['tmerg[11]'].min(), subdata['tmerg[11]'].max()
    #set bins to 50 (is there a better way to calculate it?)
    log_bins = np.logspace(np.log10(min_time),np.log10(max_time),20)    
    
    for simul_num in fMT:
        
        subsubdata = subdata[subdata['fMT[20]'] == simul_num]

        subset_CE = subsubdata[subsubdata['N_CEs[19]'] >= 1]

        subset_NOT_CE = subsubdata[subsubdata['N_CEs[19]'] == 0]


         #plot times
        sns.distplot(subset_CE['tmerg[11]'], bins =  log_bins, kde = False , ax = axtimes[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

        sns.distplot(subset_NOT_CE['tmerg[11]'], bins =  log_bins, kde = False , label = "fMT "+simul_num, ax = axtimes[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

    axtimes[0].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[0].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[0].set_title("Systems - CE phase", fontsize = 'xx-large')
    axtimes[0].set_xscale('log')
    axtimes[0].set_yscale('log')
    axtimes[0].tick_params( labelsize = 'xx-large' )


    axtimes[1].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[1].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[1].set_title("Systems - NOT CE phase", fontsize = 'xx-large')
    axtimes[1].set_xscale('log')
    axtimes[1].set_yscale('log')
    axtimes[1].tick_params( labelsize = 'xx-large' )

    fig_times.suptitle("DT distribution for Z = "+metallicity, fontsize = 'xx-large')

    handles, labels = axtimes[1].get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    fig_times.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    filename = '/DT_Z'+metallicity+'.png'
    fig_times.savefig(output_dir+filename) 
    print("Finished plots with fixed Z = ", metallicity)
    plt.close()
    

def plot_DT_fixedZ_CE(dataframe, metallicity):
        
    output_dir = "BNS/assignment02/CE_fixedZ/count_CE"
    mkdir_p(output_dir)
    
    #make a color palette of 7 colors
    sns.set_palette('inferno', 7)
    
    #initialize space for plots:
    #1 row and 2 columns
    fig_times, axtimes    = plt.subplots(1,4,figsize = (60,12), sharey = True)    
    
    subdata = dataframe[dataframe['metallicity[21]'] == metallicity]
    
    #now set the log binning set on the min and the max 
    min_time, max_time = subdata['tmerg[11]'].min(), subdata['tmerg[11]'].max()
    #set bins to 50 (is there a better way to calculate it?)
    log_bins = np.logspace(np.log10(min_time),np.log10(max_time),10)    
    
    for simul_num in fMT:
        
        subsubdata = subdata[subdata['fMT[20]'] == simul_num]

        subset_NO_CE = subsubdata[subsubdata['N_CEs[19]'] == 0]
        subset_ONE_CE = subsubdata[subsubdata['N_CEs[19]'] == 1]
        subset_TWO_CE = subsubdata[subsubdata['N_CEs[19]'] == 2]
        subset_THREE_CE = subsubdata[subsubdata['N_CEs[19]'] == 3]


         #plot times
        sns.distplot(subset_NO_CE['tmerg[11]'], bins =  log_bins, kde = False , ax = axtimes[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 
        
        sns.distplot(subset_ONE_CE['tmerg[11]'], bins =  log_bins, kde = False , ax = axtimes[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

        sns.distplot(subset_TWO_CE['tmerg[11]'], bins =  log_bins, kde = False , label = "fMT "+simul_num, ax = axtimes[2], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 
        
        sns.distplot(subset_THREE_CE['tmerg[11]'], bins =  log_bins, kde = False , label = "fMT "+simul_num, ax = axtimes[3], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

    handles, labels = axtimes[2].get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
        
    axtimes[0].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[0].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[0].set_title("Systems - NO CE phase", fontsize = 'xx-large')
    axtimes[0].set_xscale('log')
    axtimes[0].set_yscale('log')
    axtimes[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    axtimes[0].tick_params( labelsize = 'xx-large' )


    axtimes[1].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[1].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[1].set_title("Systems - 1 CE phase", fontsize = 'xx-large')
    axtimes[1].set_xscale('log')
    axtimes[1].set_yscale('log')
    axtimes[1].tick_params( labelsize = 'xx-large' )
    axtimes[1].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    axtimes[2].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[2].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[2].set_title("Systems - 2 CE phase", fontsize = 'xx-large')
    axtimes[2].set_xscale('log')
    axtimes[2].set_yscale('log')
    axtimes[2].tick_params( labelsize = 'xx-large' )
    axtimes[2].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    fig_times.suptitle("DT distribution for Z = "+metallicity, fontsize = 'xx-large')
    
    axtimes[3].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[3].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[3].set_title("Systems - with 3 CE phase", fontsize = 'xx-large')
    axtimes[3].set_xscale('log')
    axtimes[3].set_yscale('log')
    axtimes[3].tick_params( labelsize = 'xx-large' )
    axtimes[3].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')




    filename = '/DT_Z'+metallicity+'.png'
    fig_times.savefig(output_dir+filename) 
    print("Finished plots with fixed Z = ", metallicity)
    plt.close()
    
    
    
def plot_DT_fixedfMT_CE_flag(dataframe, simul_num):
    
    output_dir = "BNS/assignment02/CE_fixedfMT/CEflag"
    mkdir_p(output_dir)
    
    #make a color palette of 12 colors
    sns.set_palette('inferno', 12)
    #initialize space for plots:
    #1 row and 2 columns
    fig_times, axtimes    = plt.subplots(1,2,figsize = (30,12), sharey = True)
    
    subdata = dataframe[dataframe['fMT[20]'] == simul_num]
    
    #now set the log binning set on the min and the max 
    min_time, max_time = subdata['tmerg[11]'].min(), subdata['tmerg[11]'].max()
    #set bins to 50 (is there a better way to calculate it?)
    log_bins = np.logspace(np.log10(min_time),np.log10(max_time),20)    

    for metallicity in metallicities:
        
        subsubdata = subdata[subdata['metallicity[21]'] == metallicity]

        subset_CE = subsubdata[subsubdata['N_CEs[19]'] >= 1]
        #print(subset_CE)
        subset_NOT_CE = subsubdata[subsubdata['N_CEs[19]'] == 0]
        #print(subset_NOT_CE)

        #plot times
        sns.distplot(subset_CE['tmerg[11]'], bins =  log_bins,  kde = False , ax = axtimes[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

        sns.distplot(subset_NOT_CE['tmerg[11]'], bins =  log_bins,  kde = False , label = "Z= "+metallicity, ax = axtimes[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

    axtimes[0].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[0].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[0].set_title("Systems - CE phase", fontsize = 'xx-large')
    axtimes[0].set_xscale('log')
    axtimes[0].set_yscale('log')
    axtimes[0].tick_params( labelsize = 'xx-large' )


    axtimes[1].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[1].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[1].set_title("Systems - NOT CE phase", fontsize = 'xx-large')
    axtimes[1].set_xscale('log')
    axtimes[1].set_yscale('log')
    axtimes[1].tick_params( labelsize = 'xx-large' )

    fig_times.suptitle("DT distribution for fMT = "+simul_num, fontsize = 'xx-large')

    handles, labels = axtimes[1].get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    fig_times.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    filename = '/fmt'+simul_num+'.png'
    fig_times.savefig(output_dir+filename) 
    print("Finished plots with fixed fMT = ", simul_num)
        
    plt.close()

    
def plot_DT_fixedfMT_CE(dataframe, simul_num):
    
    output_dir = "BNS/assignment02/CE_fixedfMT/count_CE"
    mkdir_p(output_dir)
    
    #make a color palette of 12 colors
    sns.set_palette('inferno', 12)
    #initialize space for plots:
    #1 row and 2 columns
    fig_times, axtimes    = plt.subplots(1,4,figsize = (60,12), sharey = True)
    
    subdata = dataframe[dataframe['fMT[20]'] == simul_num]
    
    #now set the log binning set on the min and the max 
    min_time, max_time = subdata['tmerg[11]'].min(), subdata['tmerg[11]'].max()
    #set bins to 50 (is there a better way to calculate it?)
    log_bins = np.logspace(np.log10(min_time),np.log10(max_time),10)    

    for metallicity in metallicities:
        
        subsubdata = subdata[subdata['metallicity[21]'] == metallicity]

        subset_NO_CE = subsubdata[subsubdata['N_CEs[19]'] == 0]
        subset_ONE_CE = subsubdata[subsubdata['N_CEs[19]'] == 1]
        subset_TWO_CE = subsubdata[subsubdata['N_CEs[19]'] == 2]
        subset_THREE_CE = subsubdata[subsubdata['N_CEs[19]'] == 3]

        #plot times
        sns.distplot(subset_NO_CE['tmerg[11]'], bins =  log_bins,  kde = False , ax = axtimes[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

        sns.distplot(subset_ONE_CE['tmerg[11]'], bins =  log_bins,  kde = False , label = "Z= "+metallicity, ax = axtimes[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 
        
        sns.distplot(subset_TWO_CE['tmerg[11]'], bins =  log_bins,  kde = False , label = "Z= "+metallicity, ax = axtimes[2], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 
        
        sns.distplot(subset_THREE_CE['tmerg[11]'], bins =  log_bins,  kde = False , label = "Z= "+metallicity, ax = axtimes[3], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True) 

    handles, labels = axtimes[2].get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    
    axtimes[0].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[0].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[0].set_title("Systems - with NO CE phase", fontsize = 'xx-large')
    axtimes[0].set_xscale('log')
    axtimes[0].set_yscale('log')
    axtimes[0].tick_params( labelsize = 'xx-large' )
    axtimes[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')


    axtimes[1].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[1].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[1].set_title("Systems - with 1 CE phase", fontsize = 'xx-large')
    axtimes[1].set_xscale('log')
    axtimes[1].set_yscale('log')
    axtimes[1].tick_params( labelsize = 'xx-large' )
    axtimes[1].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    axtimes[2].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[2].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[2].set_title("Systems - with 2 CE phase", fontsize = 'xx-large')
    axtimes[2].set_xscale('log')
    axtimes[2].set_yscale('log')
    axtimes[2].tick_params( labelsize = 'xx-large' )
    axtimes[2].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    axtimes[3].set_xlabel("time [Myr]", fontsize = 'xx-large')
    axtimes[3].set_ylabel("Density", fontsize = 'xx-large')
    axtimes[3].set_title("Systems - with 3 CE phase", fontsize = 'xx-large')
    axtimes[3].set_xscale('log')
    axtimes[3].set_yscale('log')
    axtimes[3].tick_params( labelsize = 'xx-large' )
    axtimes[3].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')


    fig_times.suptitle("DT distribution for fMT = "+simul_num, fontsize = 'xx-large')



    filename = '/fmt'+simul_num+'.png'
    fig_times.savefig(output_dir+filename) 
    print("Finished plots with fixed fMT = ", simul_num)
        
    plt.close()    
    
    
    
    
    
def import_single_chunk_givenIDs_mergers(fMT, metallicity, chunk, dataset_info):
        
    filename = '../modB/simulations_fMT'+fMT+'/A5/'+metallicity+'/chunk'+chunk+'/mergers.out'
    df = pd.read_csv(filename, delim_whitespace = True, header = 0)
    df_modified = df[df.columns[:-1]]
    df_modified.columns = df.columns[1:]
    
    IDs_flags = subindices.loc[:,'ID'].unique()
    
    #this function keeps all the rows with certain IDs
    #i.e. the ones in mergers file for the BNS systems
    df_subset_IDs = df_modified[df_modified.iloc[:,0].isin(IDs_flags)]

    #insert in the returning dataset the information about whether it has done CE
    #we insert it as a list so it ignores index of the dataframe
    
    values_CE = np.sum(subindices[['first_CE', 'second_CE', 'third_CE']], axis = 1)
    df_subset_IDs.insert(18, 'N_CEs[19]', list(values_CE) )
    
    #insert now metallicity and fMT simulation value
    df_subset_IDs.insert(19, 'fMT[20]', fMT)
    df_subset_IDs.insert(20, 'metallicity[21]', metallicity)
    
    return df_subset_IDs


def a_coefficient_CEflag(dataframe, nbins = 20):
    fMT_csv   = ['0.1','0.2','0.3','0.4','0.5','0.7','1']
    col_names =  ['a', 'err_a' , 'Z', 'fMT', 'CE' ]
    results   = pd.DataFrame(columns = col_names)
    
    YES_CE_subset_DataFrame =  dataframe[dataframe['N_CEs[19]'] >= 1]
    NO_CE_subset_DataFrame  =  dataframe[dataframe['N_CEs[19]'] == 0]
    
    list_dataf = [YES_CE_subset_DataFrame, NO_CE_subset_DataFrame]
    CE_flag    = ['YES', 'NO']
    
    for subset_DataFrame, was_CE in zip(list_dataf , CE_flag):
        
        #subset for fMT
        for sim_numb, fMT_to_csv in zip( fMT, fMT_csv ):
            subset_dataframe = subset_DataFrame[subset_DataFrame['fMT[20]'] == sim_numb]

            for Z in metallicities:
                start_time = time.time()

                subsubset_dataframe = subset_dataframe[subset_dataframe['metallicity[21]'] == Z]

                subsubset_dataframe = subsubset_dataframe[(subsubset_dataframe['tmerg[11]'] > 1e2) & (subsubset_dataframe['tmerg[11]'] < 1e4)]

                #now set the log binning set on the min and the max 
                min_time, max_time = subsubset_dataframe['tmerg[11]'].min(), subsubset_dataframe['tmerg[11]'].max()
                #set bins to 50 (is there a better way to calculate it?)
                log_bins_interp = np.logspace(np.log10(min_time),np.log10(max_time), nbins)

                histog, edges = np.histogram( subsubset_dataframe['tmerg[11]'], bins=log_bins_interp , density=True)

                diff = (np.log(edges[1]) - np.log(edges[0]))
                x = [ (diff*(i+1)+0.5*diff) for i in range(len(edges)-1)]
                y = np.log(histog)

                model, V = np.polyfit(x, y, 1, cov=True)

                fit_result = {'a': model[0], 'err_a': np.sqrt(V[0][0]) , 'Z' : Z, 'fMT' : fMT_to_csv, 'CE' : was_CE  }

                results = results.append(fit_result, ignore_index = True)
                final_time = time.time()
                print("Finished for fMT", sim_numb , "  Z", Z, "    with time:", final_time - start_time )
                
    return results

def a_coefficient_CEcount(dataframe, nbins = 20):
    fMT_csv   = ['0.1','0.2','0.3','0.4','0.5','0.7','1']
    col_names =  ['a', 'err_a' , 'Z', 'fMT', 'CE' ]
    results   = pd.DataFrame(columns = col_names)
    
    for CE in dataframe['N_CEs[19]'].unique():
        subset_DataFrame =  dataframe[dataframe['N_CEs[19]'] == CE]
    
        #subset for fMT
        for sim_numb, fMT_to_csv in zip( fMT, fMT_csv ) :
            subset_dataframe = subset_DataFrame[subset_DataFrame['fMT[20]'] == sim_numb]
            for Z in metallicities[:-3]:
                start_time = time.time()

                subsubset_dataframe = subset_dataframe[subset_dataframe['metallicity[21]'] == Z]

                subsubset_dataframe = subsubset_dataframe[(subsubset_dataframe['tmerg[11]'] > 1e2) & (subsubset_dataframe['tmerg[11]'] < 1e4)]

                #now set the log binning set on the min and the max 
                min_time, max_time = subsubset_dataframe['tmerg[11]'].min(), subsubset_dataframe['tmerg[11]'].max()
                #set bins to 50 (is there a better way to calculate it?)
                log_bins_interp = np.logspace(np.log10(min_time),np.log10(max_time), nbins)

                histog, edges = np.histogram( subsubset_dataframe['tmerg[11]'], bins=log_bins_interp , density=True)

                diff = (np.log(edges[1]) - np.log(edges[0]))
                x = [ (diff*(i+1)+0.5*diff) for i in range(len(edges)-1)]
                y = np.log(histog)

                model, V = np.polyfit(x, y, 1, cov=True)

                fit_result = {'a': model[0], 'err_a': np.sqrt(V[0][0]) , 'Z' : Z, 'fMT' : fMT_to_csv, 'CE' : CE  }
                print(fit_result)

                results = results.append(fit_result, ignore_index = True)
                results.to_csv('a_coefficient_CEcount.csv', index=False)

                final_time = time.time()
                print("Finished for fMT", sim_numb , "  Z", Z, "    with time:", final_time - start_time )
    return results
            

#fMT = ['01','02','03','04','05','07','1'] 
#metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
#chunks = ['0','1','2','3','4']

fMT_csv   = ['0.1','0.2','0.3','0.4','0.5','0.7','1']



dataf_indexes = import_database_ID_nCEs()

fMT           = dataf_indexes.loc[:,'fMT'].unique()
metallicities = dataf_indexes.loc[:,'Z'].unique()
chunks        = dataf_indexes.loc[:,'chunk'].unique()


dataf = []
for sim_num, fMT_to_csv in zip(fMT, fMT_csv):
    list_temp_dataframesB = []
    for Z in metallicities:
        
        start_time = time.time()
        list_temp_dataframesA = []
        
        for chunk in chunks:
            
            start_chunk = time.time()
            
            print("Importing chunk", chunk, "of fMT", sim_num , "Z", Z)
            
            subindices = dataf_indexes[(dataf_indexes.loc[:,'chunk'] == chunk)&(dataf_indexes.loc[:,'Z'] == Z )&(dataf_indexes.loc[:,'fMT'] == sim_num)]
            
            data_temp = import_single_chunk_givenIDs_mergers(sim_num, Z, chunk, subindices)
            
            end_chunk = time.time()
            
            print("Importing chunk", chunk, "of fMT", sim_num , "Z", Z, "with time:", end_chunk - start_chunk )
            list_temp_dataframesA.append(data_temp)
            
        dataf_givenZ_givenFMT = pd.concat(list_temp_dataframesA, ignore_index = True)    
        list_temp_dataframesB.append(dataf_givenZ_givenFMT)
    
    dataf_givenFMT = pd.concat( list_temp_dataframesB, ignore_index = True) 
    dataf.append(dataf_givenFMT)
    
total_dataframe = pd.concat(dataf, ignore_index = True)
final_time = time.time()
print("Imported all the dataframe with time:", final_time - start_time, "\n" )

# total_dataframe.to_csv('mergers_ID_CE.csv', index=False)
# print(total_dataframe.loc[:,'N_CEs[19]'].unique())

# print("Beginning with plots...", "\n" )

# start_time = time.time()

# for simul_num in fMT:
#     plot_DT_fixedfMT_CE_flag(total_dataframe, simul_num)
#     plot_DT_fixedfMT_CE(total_dataframe, simul_num)
#     plt.close()
    
# for Z in metallicities:
#     plot_DT_fixedZ_CE_flag(total_dataframe, Z)
#     plot_DT_fixedZ_CE(total_dataframe, Z)
#     plt.close()
        
# final_time = time.time()

# print("Done with plots with time:", final_time - start_time, "\n" )

start_time = time.time()
print("Now starting computing the a coefficient for CE flag...")           
result = a_coefficient_CEflag(total_dataframe, 20)
result.to_csv('a_coefficient_CEflag.csv', index=False)
final_time = time.time()
print("Done with computing a coefficient with time:", final_time - start_time, "\n" )            

start_time = time.time()
print("Now starting computing the a coefficient for CE count...")           
result = a_coefficient_CEcount(total_dataframe, 6)
result.to_csv('a_coefficient_CEcount.csv', index=False)
final_time = time.time()
print("Done with computing a coefficient with time:", final_time - start_time, "\n" )    












