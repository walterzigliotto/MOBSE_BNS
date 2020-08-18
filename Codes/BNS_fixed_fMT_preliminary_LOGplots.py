import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   

# r=d * (m1/m2)**(1./3) 
# galactic dynamics by binney & tremaine 
# r=d * (m1/m2)**(1./3) 
# r=d * (mNS/mBH)**(1./3) 

fMT = ['01','02','03','04','05','07','1']
metallicities = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunks = ['0','1','2','3','4']

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


def plot_assignment1(dataframe):
    
    #make a color palette of 12 colors
    sns.set_palette('inferno', 12)

    #dataframe is the dataframe with included metallicities and so forth
    #here the code to make ONE graph for each fMT and keep all the metallicities
    #could be done better? yes. Will I do it? Maybe <3

    list_values_met = dataframe.metallicity.unique()
    list_values_fmT = dataframe.fMT.unique()
    sim_numb = list_values_fmT[0]

    #initialize space for plots
    fig_m1, axm1         = plt.subplots(figsize = (18,12))
    fig_m2, axm2         = plt.subplots(figsize = (18,12))
    fig_mtot, axmtot     = plt.subplots(figsize = (18,12))
    fig_q, axq           = plt.subplots(figsize = (18,12))
    fig_mchirp, axmchirp = plt.subplots(figsize = (18,12))

    #it returns the metallicities of the list above, but actually code is more flexible
    for metallicity in list_values_met:

        #retrieve only a single metallicity and the string of fmT value
        subset_data = dataframe[dataframe['metallicity'] == metallicity]
        simul_num   = dataframe.iloc[0]['fMT']

        #DATA PREPARATION
        m_tot   = subset_data['m1form[8]']+subset_data['m2form[10]']
        q       = subset_data['m2form[10]']/subset_data['m1form[8]']
        m_chirp = (subset_data['m2form[10]']/subset_data['m1form[8]'])**(3/5.)/(subset_data['m1form[8]']+subset_data['m2form[10]'])**(1/5.)


        #PLOTTING STUFF
        #first m1 plot, ax=axm1 is to superimpose to an unique plot the all different metallicities
        sns.distplot(subset_data['m1form[8]'], kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axm1, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot m2
        sns.distplot(subset_data['m2form[10]'], kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axm2, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot mtot
        sns.distplot(m_tot, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axmtot, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot q
        sns.distplot(q, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axq, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot m_chirp
        sns.distplot(m_chirp, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axmchirp, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

    axm1.set_xlabel("m1 $[M_\odot]$", fontsize = 'xx-large')
    axm1.set_ylabel("Density", fontsize = 'xx-large')
    axm1.set_title("Distribution of m1 for Z = "+metallicity, fontsize = 'xx-large')
    axm1.set_yscale('log')
    
    axm2.set_xlabel("m2 $[M_\odot]$", fontsize = 'xx-large')
    axm2.set_ylabel("Density", fontsize = 'xx-large')
    axm2.set_title("Distribution of m2 for Z"+metallicity, fontsize = 'xx-large')
    axm2.set_yscale('log')
    
    axmtot.set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 'xx-large')
    axmtot.set_ylabel("Density", fontsize = 'xx-large')
    axmtot.set_title("Distribution of $m_{tot} = m_1+m_2$ for Z = "+metallicity, fontsize = 'xx-large')
    axmtot.set_yscale('log')

    axq.set_xlabel("q", fontsize = 'xx-large')
    axq.set_ylabel("Density", fontsize = 'xx-large')
    axq.set_title("Distribution of $q= m_2 / m_1$ for Z = "+metallicity, fontsize = 'xx-large')
    axq.set_yscale('log')
    
    axmchirp.set_xlabel("$m_{chirp}$ $[M_\odot]$", fontsize = 'xx-large')
    axmchirp.set_ylabel("Density", fontsize = 'xx-large')
    axmchirp.set_title("Distribution of $m_{chirp}$ for Z = "+metallicity, fontsize = 'xx-large')
    axmchirp.set_yscale('log')

    #THIS CODE IS TO AVOID THAT IN THE LEGEND THERE IS A VOID SQUARE AND NOT A LINE (see the notebook I sent you)
    #once I figure out how to sample at the center of each histogram I think I'll drop this out because it is
    #more aesthetic and kawaii
    handles, labels = axm1.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm1.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    handles, labels = axm2.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm2.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    handles, labels = axmtot.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmtot.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    handles, labels = axq.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axq.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    handles, labels = axmchirp.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmchirp.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    
    axm1.tick_params( labelsize = 'xx-large' )
    axm2.tick_params( labelsize = 'xx-large' )
    axmtot.tick_params( labelsize = 'xx-large' )
    axq.tick_params( labelsize = 'xx-large' )
    axmchirp.tick_params( labelsize = 'xx-large' )

    # Create new directory
    output_dir = "BNS/fixed_fMT/fMT"+sim_numb+"/logplots"
    mkdir_p(output_dir)

    fig_m1.savefig('{}/m1.png'.format(output_dir))
    fig_m2.savefig('{}/m2.png'.format(output_dir))
    fig_mtot.savefig('{}/mtot.png'.format(output_dir))
    fig_q.savefig('{}/q.png'.format(output_dir))
    fig_mchirp.savefig('{}/mchirp.png'.format(output_dir))

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

    data = pd.concat(temp_dataframesB, ignore_index=True)
    plot_assignment1(data)
