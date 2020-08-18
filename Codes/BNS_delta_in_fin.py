import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


def m_in_m_fin_diff_1(dataframe):
    m_in = dataframe['min1[2]']
    m_fin = dataframe['m1form[8]']
    return(m_in-m_fin)

def m_in_m_fin_diff_2(dataframe):
    m_in = dataframe['min2[3]']
    m_fin = dataframe['m2form[10]']
    return(m_in-m_fin)

fMT = ['01','02','03','04','05','07','1']
metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunks = ['0','1','2','3','4']
fMT_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.7', '1']


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



def plot_minVmfin(dataframe):

    output_dir = "prov/"
    mkdir_p(output_dir)

    sns.set_palette('inferno', 12)

    ##############
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10), sharey= True)



    subdatatotal1 = datatotal[datatotal['fMT'] == '01']

    #to change
    subdatatotal2 = datatotal[datatotal['fMT'] == '1']


    for metal in metallicities:

        subdatatotal4 = subdatatotal1[subdatatotal1['metallicity'] == metal]
        subdatatotal5 = subdatatotal2[subdatatotal2['metallicity'] == metal]

        sns.scatterplot(x=m_in_m_fin_diff_1(subdatatotal4), y=m_in_m_fin_diff_2(subdatatotal4), alpha = 0.3,
               label = "Z = "+str(metal), ax = ax[0])

        sns.scatterplot(x=m_in_m_fin_diff_1(subdatatotal5), y=m_in_m_fin_diff_2(subdatatotal5), alpha = 0.3,
               label = "Z = "+str(metal), ax = ax[1])

    ax[0].set_ylabel("$\Delta m_2$ $[M_\odot]$", fontsize = 'xx-large')
    ax[0].set_xlabel("$\Delta m_1$ $[M_\odot]$", fontsize = 'xx-large')
    ax[0].legend(loc="upper left", title="Metallicity", fontsize = 'x-large', title_fontsize = 'xx-large')
    ax[0].set_title("fMT = 0.1", fontsize = 'xx-large')            
    ax[0].tick_params(labelsize = 'x-large')


    ax[1].set_xlabel("$\Delta m_1$ $[M_\odot]$", fontsize = 'xx-large')
    ax[1].legend(loc="upper left", title="Metallicity", fontsize = 'x-large', title_fontsize = 'xx-large')
    ax[1].set_title("fMT = 1", fontsize = 'xx-large')            
    ax[1].tick_params(labelsize = 'x-large')

    fig.suptitle('BNSs and ZAMSs mass differences in conservative and non consevative cases', fontsize = 22)



        
    filename = 'ZAMS_vs_BNSs.png'
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

plot_minVmfin(datatotal)
final_time = time.time()
print("Total time execution: ", final_time - start_time )
