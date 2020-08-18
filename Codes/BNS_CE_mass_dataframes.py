import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.lines import Line2D   
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


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

def import_single_chunk_givenIDs_evol_mergers(fMT, metallicity, chunk, dataset_info):
        
    filename = '../modB/simulations_fMT'+fMT+'/A5/'+metallicity+'/chunk'+chunk+'/evol_mergers.out'
    df = pd.read_csv(filename, delim_whitespace = True, header = 0)
    df_modified = df[df.columns[:-1]]
    df_modified.columns = df.columns[1:]
    
    IDs_flags = subindices.loc[:,'ID'].unique()
    
    #this function keeps all the rows with certain IDs
    #i.e. the ones in mergers file for the BNS systems
    df_subset_IDs = df_modified[df_modified.iloc[:,0].isin(IDs_flags)]

    #insert in the returning dataset the information about whether it has done CE
    #we insert it as a list so it ignores index of the dataframe
    
    count_CE = list(np.sum(dataset_info.iloc[:,4:9].astype(bool), axis = 1)) 
    
    #take the initial masses, i.e. the ones with label INITIAL
    initial_df = df_subset_IDs[df_subset_IDs['label[33]'] == 'INITIAL'].loc[:,['mt1[4]','mt2[18]']]
    
    #take final masses, i.e. the first ones for which k1 and k2 are 13
    #first choose events where both objects are 13, then group them by ID, take first element, forget about the index,
    #and finally save info we are interested in
    final_df   = df_subset_IDs[(df_modified['k1[2]'] == 13) & (df_modified['k2[16]'] == 13)].groupby('ID[0]').first().reset_index().loc[:,['mt1[4]','mt2[18]']]

    initial_df.insert(2, 'N_CEs', count_CE)
    initial_df.insert(3, 'fMT', fMT)
    initial_df.insert(4, 'metallicity', metallicity)
    
    final_df.insert(2, 'N_CEs', count_CE)
    final_df.insert(3, 'fMT', fMT)
    final_df.insert(4, 'metallicity', metallicity)

    
    if len(final_df) != len(initial_df): print("Different lengths!", fMT, metallicity, chunk)
        
    return initial_df, final_df



def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs ):
    sns.set_palette('inferno')

    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs, cmap = 'inferno' )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = plt.colorbar(cm.ScalarMappable(norm = norm,   cmap = 'inferno'), ax=ax)
    cbar.ax.set_ylabel('Density', fontsize = 12)
    cbar.ax.tick_params(labelsize = 10)


    return ax





def scatter_plot_masses(dataframe, title):
    output_dir = "BNS/assignment02/CE_plots/"
    mkdir_p(output_dir)
    
    sns.set_palette('inferno')
    
    fig_mass, mass    = plt.subplots(1,3,figsize = (24,6), sharey = True)
        
    subset_ONE_CE   = dataframe[dataframe['N_CEs'] == 1]
    subset_TWO_CE   = dataframe[dataframe['N_CEs'] == 2]
    subset_THREE_CE = dataframe[dataframe['N_CEs'] == 3]
    
    density_scatter(np.array(subset_ONE_CE.loc[:,'mt1[4]'])  , np.array(subset_ONE_CE.loc[:,'mt2[18]'])  , ax = mass[0])
    density_scatter(np.array(subset_TWO_CE.loc[:,'mt1[4]'])  , np.array(subset_TWO_CE.loc[:,'mt2[18]'])  , ax = mass[1])
    density_scatter(np.array(subset_THREE_CE.loc[:,'mt1[4]']), np.array(subset_THREE_CE.loc[:,'mt2[18]']), ax = mass[2])
    
    mass[0].set_xlabel("m1  $[M_\odot]$", fontsize = 'xx-large')
    mass[0].set_ylabel("m2  $[M_\odot]$", fontsize = 'xx-large')
    mass[0].set_title("ONE CE", fontsize = 'xx-large')
    mass[0].tick_params( labelsize = 'xx-large' )
    
    mass[1].set_xlabel("m1  $[M_\odot]$", fontsize = 'xx-large')
    mass[1].set_ylabel("m2  $[M_\odot]$", fontsize = 'xx-large')
    mass[1].set_title("TWO CE", fontsize = 'xx-large')
    mass[1].tick_params( labelsize = 'xx-large' )
    
    
    mass[2].set_xlabel("m1  $[M_\odot]$", fontsize = 'xx-large')
    mass[2].set_ylabel("m2  $[M_\odot]$", fontsize = 'xx-large')
    mass[2].set_title("THREE CE", fontsize = 'xx-large')
    mass[2].tick_params( labelsize = 'xx-large' )
    
    fig_mass.suptitle(title, fontsize = 25)

    filename = title+'_scatter.png'
    fig_mass.savefig(output_dir+filename) 

    
    
def q(dataframe):
    
    m1 = dataframe['mt1[4]']
    m2 = dataframe['mt2[18]']
    
    vector = []
    
    for first, second in zip(m1,m2):
        ratio = second/first
        
        if    ratio <= 1: vector.append(ratio)
        elif  ratio > 1 : vector.append(np.power(ratio,-1))
        
    return(np.array(vector))


fMT = ['01','02','03','04','05','07','1']
fMT_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.7', '1']
fMT_csv   = ['0.1','0.2','0.3','0.4','0.5','0.7','1']

metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunks = ['0','1','2','3','4']




# fMT_csv   = ['0.1']
# fMT = ['01']
# fMT_labels = ['0.1']
# metallicities = ['0.0002']

dataf_indexes = import_database_ID_nCEs()


dataf_init = []
dataf_fin  = []
for sim_num, fMT_to_csv in zip(fMT, fMT_csv):
    list_temp_dataframesB_init = []
    list_temp_dataframesB_fin  = []
    
    for Z in metallicities:
        
        start_time = time.time()
        list_temp_dataframesA_init = []
        list_temp_dataframesA_fin = []
        
        for chunk in chunks:
            
            start_chunk = time.time()
            
            print("Importing chunk", chunk, "of fMT", sim_num , "Z", Z)
            
            subindices = dataf_indexes[(dataf_indexes.loc[:,'chunk'] == chunk)&(dataf_indexes.loc[:,'Z'] == Z )&(dataf_indexes.loc[:,'fMT'] == sim_num)]
            
            data_temp_init, data_temp_fin = import_single_chunk_givenIDs_evol_mergers(sim_num, Z, chunk, subindices)
            
            end_chunk = time.time()
            
            print("Importing chunk", chunk, "of fMT", sim_num , "Z", Z, "with time:", end_chunk - start_chunk )
            list_temp_dataframesA_init.append(data_temp_init)
            list_temp_dataframesA_fin.append(data_temp_fin)

            
        dataf_givenZ_givenFMT_init = pd.concat(list_temp_dataframesA_init, ignore_index = True)  
        dataf_givenZ_givenFMT_fin  = pd.concat(list_temp_dataframesA_fin,  ignore_index = True)  

        
        
        list_temp_dataframesB_init.append(dataf_givenZ_givenFMT_init)
        list_temp_dataframesB_fin.append(dataf_givenZ_givenFMT_fin)

        
    
    dataf_givenFMT_init = pd.concat( list_temp_dataframesB_init, ignore_index = True) 
    dataf_givenFMT_fin  = pd.concat( list_temp_dataframesB_fin, ignore_index = True) 

    
    dataf_init.append(dataf_givenFMT_init)
    dataf_fin.append(dataf_givenFMT_fin)
    
    
total_dataframe_init = pd.concat(dataf_init, ignore_index = True)
total_dataframe_fin = pd.concat(dataf_fin, ignore_index = True)


total_dataframe_init.to_csv('BNS/assignment02/CE_masses_initial.csv', index=False)
total_dataframe_fin.to_csv('BNS/assignment02/CE_masses_final.csv', index=False)



final_time = time.time()
print("Imported all the dataframe with time:", final_time - start_time, "\n" )

# print("Beginning with plots...")
# plot_time_init_start = time.time()
# scatter_plot_masses(total_dataframe_init, "Initial masses")
# plot_time_init_end = time.time()
# print("Time passed", plot_time_init_end - plot_time_init_start)


# plot_time_fin_start = time.time()
# scatter_plot_masses(total_dataframe_fin,  "Final masses")
# plot_time_fin_end = time.time()
# print("Time passed", plot_time_fin_end - plot_time_fin_start)


