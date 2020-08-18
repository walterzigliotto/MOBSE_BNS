import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time



def m_chirp(dataframe):
    ratio = q(dataframe)
    mtot = m_tot(dataframe)
    return (ratio)**(3/5.)/(mtot)**(1/5.)

def q(dataframe):
    m1 = dataframe['m1form[8]']
    m2 = dataframe['m2form[10]']
    return(m2/m1)

def m_tot(dataframe):
    m1 = dataframe['m1form[8]']
    m2 = dataframe['m2form[10]']
    return(m1+m2)

fMT = ['01','02','03','04','05','07','1']
metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunks = ['0','1','2','3','4']
fMT_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.7', '1']

fMT = ['05']
fMT_labels = ['0.5']

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



def plot_eccVSsep(dataframe):

    output_dir = "BNS/ecc_vs_sep/"
    mkdir_p(output_dir)
    pointsize = 10


    #subset for fMT
    for sim_numb, fmt_lab in zip(fMT, fMT_labels):
        subset_dataframe = dataframe[dataframe['fMT'] == sim_numb]
        
        fig, ax = plt.subplots(nrows = 3, ncols = 4, figsize = (20, 16), sharex = True, sharey = True)
        index = 0
        
        for Z in metallicities:
            
            subsubset_dataframe = subset_dataframe[subset_dataframe['metallicity'] == Z]

            im = ax[index//4,index%4].scatter( subsubset_dataframe['eccform[6]'],               #x
                       subsubset_dataframe['sepform[5]'],                                       #y in log scale!
                       pointsize,
                       np.log(subsubset_dataframe['tmerg[11]']),                                #different colors
                       cmap = 'inferno')
            
            ax[index//4,index%4].set_ylabel("separation", fontsize = 'xx-large')
            ax[index//4,index%4].set_xlabel('eccentricity', fontsize = 'xx-large')            
            ax[index//4,index%4].set_title("Z = "+Z, fontsize = 'xx-large')            
            ax[index//4,index%4].set_yscale('log')
            ax[index//4,index%4].tick_params(labelsize = 'x-large')
            index += 1
    
#         fig.suptitle("log(delay time) in function of [log(distance), eccentricity] for fMT="+sim_numb, fontsize = 'xx-large')
        fig.suptitle("fMT = "+fmt_lab, fontsize = 40)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label( label = 'log(delay time)',fontsize = 'xx-large')
        cbar.ax.tick_params(labelsize = 'x-large')
        
        filename = 'fMT_'+sim_numb+'.png'
        fig.savefig(output_dir+filename)


data_list = []

start_time = time.time()

for i in fMT:
    temp_dataframesB = []
    for metallicity in metallicities:

        temp_dataframesA = []

        for chunk in chunks:
            filename = '../modB/simulations_fMT'+i+'/A5/'+metallicity+'/chunk'+chunk+'/mergers.out'

            df = pd.read_csv(filename, delim_whitespace = True, header = 0)
            df_modified = df[df.columns[:-1]]
            df_modified.columns = df.columns[1:]

            #we want to extract NB systems
            df_subset = df_modified[(df_modified['k1form[7]'] == 13) & (df_modified['k2form[9]'] == 13 ) ]
            df_subset['metallicity'] = metallicity
            df_subset['fMT'] = i

            temp_dataframesA.append(df_subset)

        #the next one is a dataframe containing all a given metallicity
        data_givenZ = pd.concat(temp_dataframesA, ignore_index=True)
        temp_dataframesB.append(data_givenZ)

    data_list.append(pd.concat(temp_dataframesB, ignore_index=True))
    
middle_time = time.time()
print("\n\n\n\nTime passed while importing all chunks:", middle_time - start_time )
print("Now creating data total dataframe!")
#create a dataframe with whole data inside it
datatotal = pd.concat(data_list, ignore_index=True)
middle_time = time.time()

print("Time passed creating the whole data dataframe: ", middle_time - start_time )

plot_eccVSsep(datatotal)
final_time = time.time()
print("Total time execution: ", final_time - start_time )
