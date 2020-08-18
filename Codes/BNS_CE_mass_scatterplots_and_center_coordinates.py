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

def import_database(filename):
    
    path = 'BNS/assignment02/'
    df = pd.read_csv(path+filename, header = 0, dtype = {'fMT': object, 'Z' : object, 'chunk': object})

    return df


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





def scatter_plot_masses_fixedZ(data, title):
    
    sns.set_palette('inferno')
    
    fMT_values = data.loc[:,'fMT'].unique()
    Z_values   = data.loc[:,'metallicity'].unique()
    
    for metallicity in Z_values:
        
        subset_met = data[data['metallicity'] == metallicity] 
        
        for fMT in fMT_values:
        
            dataframe  = subset_met[subset_met['fMT'] == fMT]

            fig_mass, mass  = plt.subplots(1,3,figsize = (24,6), sharey = True)

            
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

            fig_mass.suptitle(title+ "   fMT = "+str(fMT), fontsize = 25)
            
            output_dir = "BNS/assignment02/CE_plots/scatter_mass/fixedZ"+str(metallicity)+"/"
            mkdir_p(output_dir)
            
            filename = str(title+'__fMT'+str(fMT)+'_scatter.png')

            fig_mass.savefig(output_dir+filename) 
            
            plt.close()
    
def scatter_plot_masses_fixedfMT(data, title):

    sns.set_palette('inferno')
    
    fMT_values = data.loc[:,'fMT'].unique()
    Z_values   = data.loc[:,'metallicity'].unique()
    
    for fMT in fMT_values:
        subset_fMT  =   data[data['fMT'] == fMT] 

        for metallicity in Z_values:
        
            dataframe = subset_fMT[subset_fMT['metallicity'] == metallicity]
        
            fig_mass, mass  = plt.subplots(1,3,figsize = (24,6), sharey = True)

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

            fig_mass.suptitle(title+ "   Z = "+str(metallicity), fontsize = 25)
            
            output_dir = "BNS/assignment02/CE_plots/scatter_mass/fixedfMT"+str(fMT)+"/"
            mkdir_p(output_dir)
            
            filename = str(title+'__Z'+str(metallicity)+'_scatter.png')
            fig_mass.savefig(output_dir + filename) 

            plt.close()
            
            
def find_most_probable_center(x, y, sort = True, bins = 30):

    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, the last one is the one with highest density
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    return x[len(x)-1], y[len(y)-1]

def mass_flip_df(data):
    
    col_names = ['fraction', 'fMT', 'Z', 'CE']
    values = pd.DataFrame(columns = col_names)
    
    fMT_values = data.loc[:,'fMT'].unique()
    Z_values   = data.loc[:,'metallicity'].unique()
    n_CEs      = [1,2,3]
    
    for fMT in fMT_values:
        subset_fMT   =   data [data['fMT']  == fMT] 

        for metallicity in Z_values:
            subset_fMT_Z   =   subset_fMT [subset_fMT['metallicity']  == metallicity] 
            
            for n_CE in n_CEs:
                subset_fMT_Z_nCE   =   subset_fMT_Z [subset_fMT_Z['N_CEs']  == n_CE]

                array_bool = subset_fMT_Z_nCE.loc[:,'mt2[18]'] > subset_fMT_Z_nCE.loc[:,'mt1[4]'] 
                                
                flip_count = np.sum(array_bool)
                values = values.append({ 'fraction' : flip_count/len(array_bool), 'fMT' : fMT, 'Z' : metallicity, 'CE' : n_CE}, ignore_index = True)
    
    return values
            
def mass_flip_plots(data):
    
    fMT_values = data.loc[:,'fMT'].unique()
    fMT_to_csv = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.7', '1.0']
    
    Z_values   = data.loc[:,'Z'].unique()
    n_CEs      = data.loc[:,'CE'].unique()
        
    sns.set_palette('inferno', 7)

    fig, ax = plt.subplots(1,3, figsize = (30,10), sharey = True)
    
    for fMT, fMT_lab in zip(fMT_values, fMT_to_csv) :
            data_fMT = data[data['fMT'] == fMT]
            
            for CE, axes in zip(n_CEs, ax.flatten()):
                data_fMT_CE = data_fMT[data_fMT['CE'] == CE ]
                sns.lineplot( data_fMT_CE['Z'].astype(float), data['fraction'].astype(float), label = str(fMT_lab), ax = axes )

    plt.suptitle('Flipped masses fraction vs Metallicity', fontsize = 30)

    ax[0].set_title('ONE CE', fontsize = 20)
    ax[0].set_xlabel('Metallicity Z', fontsize = 20)
    ax[0].legend(fontsize = 'xx-large', title = 'fMT', title_fontsize = 18)
    ax[0].set_ylabel('fraction', fontsize = 20)
    ax[0].set_xscale('log')
    ax[0].tick_params( labelsize = 20 )
    ax[0].grid(True, which="both", ls="-",color='0.93')    


    ax[1].set_xlabel('Metallicity Z', fontsize = 20)
    ax[1].set_ylabel('fraction', fontsize = 20)
    ax[1].legend(fontsize = 'xx-large', title = 'fMT', title_fontsize = 18)
    ax[1].set_title('TWO CE',fontsize = 20)
    ax[1].set_xscale('log')
    ax[1].tick_params( labelsize = 20 )
    ax[1].grid(True, which="both", ls="-",color='0.93')    


    ax[2].set_title('THREE CE', fontsize = 20)
    ax[2].set_xlabel('Metallicity Z', fontsize = 20)
    ax[2].legend(fontsize = 'xx-large', title = 'fMT', title_fontsize = 18)
    ax[2].set_ylabel('fraction', fontsize = 20)
    ax[2].set_xscale('log')
    ax[2].tick_params( labelsize = 20 )
    ax[2].grid(True, which="both", ls="-",color='0.93')    

    output_dir = "BNS/assignment02/CE_plots/"
    mkdir_p(output_dir)
    fig.savefig(output_dir + 'flipped_CE_vs_metallicity.png') 
    plt.close()


    sns.set_palette('inferno', 12)

    fig, ax = plt.subplots(1,3, figsize = (30,10), sharey = True)

    for Z in Z_values :
            data_Z = data[data['Z'] == Z]
            
            for CE, axes in zip(n_CEs, ax.flatten()):
                data_Z_CE = data_Z[data_Z['CE'] == CE ]
                sns.lineplot( data_Z_CE['fMT'], data['fraction'].astype(float), label = str(Z), ax = axes )

    plt.suptitle('flipped masses fraction vs fMT', fontsize = 30)


    ax[0].set_title('ONE CE', fontsize = 20)
    ax[0].set_xlabel('fMT', fontsize = 20)
    ax[0].set_ylabel('fraction', fontsize = 20)
    ax[0].legend(fontsize = 20, title = 'Z', title_fontsize = 18, ncol = 2)
    ax[0].tick_params( labelsize = 20 )
    ax[0].grid(True, which="both", ls="-",color='0.93')    

    
    ax[1].set_title('TWO CE', fontsize = 20)
    ax[1].set_xlabel('fMT', fontsize = 20)
    ax[1].set_ylabel('fraction', fontsize = 20)
    ax[1].legend(fontsize = 20, title = 'Z', title_fontsize = 18)
    ax[1].tick_params( labelsize = 20 )
    ax[1].grid(True, which="both", ls="-",color='0.93')    


    ax[2].set_title('THREE CE', fontsize = 20)
    ax[2].set_xlabel('fMT', fontsize = 20)
    ax[2].legend(fontsize = 20, title = 'Z', title_fontsize = 18)
    ax[2].set_ylabel('fraction', fontsize = 20)
    ax[2].tick_params( labelsize = 20 )
    ax[2].grid(True, which="both", ls="-",color='0.93')    

    fig.savefig(output_dir + 'flipped_CE_vs_fMT.png')
    plt.close()
    
      
    

def dataframe_centers(data, title):
    
    col_names =  ['x', 'y' , 'Z', 'fMT', 'CE' ]
    results   = pd.DataFrame(columns = col_names)
    
    fMT_values = data.loc[:,'fMT'].unique()
    Z_values   = data.loc[:,'metallicity'].unique()
    CE_values  = [3,2,1]
    
    for fMT in fMT_values:
        subset_fMT  =   data[data['fMT'] == fMT]
   
        for metallicity in Z_values:
            dataframe = subset_fMT[subset_fMT['metallicity'] == metallicity] 
            
            for n_CE in CE_values:
                subset = dataframe[dataframe['N_CEs'] == n_CE]
                
                x_coord, y_coord = find_most_probable_center( np.array(subset.loc[:,'mt1[4]']) , np.array(subset.loc[:,'mt2[18]']) )
                
            

                result_dictionary = {'x': x_coord, 'y': y_coord , 'Z' : metallicity, 'fMT' : fMT, 'CE' : n_CE }
                results = results.append(result_dictionary, ignore_index = True)
    
    results.to_csv('BNS/assignment02/scatter_centers'+title+'.csv', index = False)
                                                             
            
            

    

start_time = time.time()            
initial_data = import_database('CE_masses_initial.csv')
final_data   = import_database('CE_masses_final.csv')
final_time   = time.time()
print("Done with importing data: ", final_time - start_time, "\n" )

# start_time = time.time()
# scatter_plot_masses_fixedZ(initial_data, 'initial_masses')
# scatter_plot_masses_fixedZ(initial_data, 'final_masses')
# final_time   = time.time()
# print("Done with scatter plots fixed Z :", final_time - start_time, "\n" )


                                                             
# start_time = time.time()
# scatter_plot_masses_fixedfMT(initial_data, 'initial_masses')
# scatter_plot_masses_fixedfMT(final_data, 'final_masses')
# final_time   = time.time()
# print("Done with scatter plots fixed fMT :", final_time - start_time, "\n" )

                                                             
# start_time = time.time()
# dataframe_centers(initial_data , 'initial')
# dataframe_centers(final_data   , 'final')
# final_time   = time.time()
# print("Done with coordinates :", final_time - start_time, "\n" )

                       
                                                             
start_time = time.time()
values = mass_flip_df(final_data)
mass_flip_plots(values)
final_time   = time.time()
print("Done with coordinates :", final_time - start_time, "\n" )
