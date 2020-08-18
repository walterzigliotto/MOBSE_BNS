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
    fig_m1, axm1         = plt.subplots(figsize = (12,8))
    fig_m2, axm2         = plt.subplots(figsize = (12,8))
    fig_mtot, axmtot     = plt.subplots(figsize = (12,8))
    fig_q, axq           = plt.subplots(figsize = (12,8))
    fig_mchirp, axmchirp = plt.subplots(figsize = (12,8))
    fig_times, axtimes    = plt.subplots(figsize = (12,8))
    
#     fig_m1_zoom  , axm1_zoom         = plt.subplots(figsize = (18,12))
#     fig_mtot_zoom, axmtot_zoom       = plt.subplots(figsize = (18,12))
    
    #now set the log binning set on the min and the max 
    min_time, max_time = dataframe['tmerg[11]'].min(), dataframe['tmerg[11]'].max()
    #set bins to 50 (is there a better way to calculate it?)
    log_bins = np.logspace(np.log10(min_time),np.log10(max_time),50)


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
        sns.distplot(subset_data['m1form[8]'], kde = False, label = metallicity, ax = axm1, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
#         sns.distplot(subset_data['m1form[8]'], kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axm1_zoom, hist_kws={"histtype": "step"})

        #plot m2
        sns.distplot(subset_data['m2form[10]'], kde = False, label = metallicity, ax = axm2, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot mtot
        sns.distplot(m_tot, kde = False, label = metallicity, ax = axmtot, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
#         sns.distplot(m_tot, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axmtot_zoom, hist_kws={"histtype": "step"})

        #plot q
        sns.distplot(q, kde = False, label = metallicity, ax = axq, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot m_chirp
        sns.distplot(m_chirp, kde = False, label = metallicity, ax = axmchirp, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot times
        sns.distplot(subset_data['tmerg[11]'], bins = log_bins,  kde = False , label = metallicity, ax = axtimes, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

    axm1.set_xlabel("m1 $[M_\odot]$", fontsize = 'xx-large')
    axm1.set_ylabel("Density", fontsize = 'xx-large')
    axm1.set_title("Distribution of m1 for fMT"+simul_num, fontsize = 'xx-large')

    axm2.set_xlabel("m2 $[M_\odot]$", fontsize = 'xx-large')
    axm2.set_ylabel("Density", fontsize = 'xx-large')
    axm2.set_title("Distribution of m2 for fMT"+simul_num, fontsize = 'xx-large')

    axmtot.set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 'xx-large')
    axmtot.set_ylabel("Density", fontsize = 'xx-large')
    axmtot.set_title("Distribution of $m_{tot} = m_1+m_2$ for Z"+simul_num, fontsize = 'xx-large')

    axq.set_xlabel("q", fontsize = 'xx-large')
    axq.set_ylabel("Density", fontsize = 'xx-large')
    axq.set_title("Distribution of $q= m_2 / m_1$ for fMT"+simul_num, fontsize = 'xx-large')

    axmchirp.set_xlabel("$m_{chirp}$ $[M_\odot]$", fontsize = 'xx-large')
    axmchirp.set_ylabel("Density", fontsize = 'xx-large')
    axmchirp.set_title("Distribution of $m_{chirp}$ for fMT"+simul_num, fontsize = 'xx-large')
    
#     axm1_zoom.set_xlabel("m1 $[M_\odot]$", fontsize = 'xx-large')
#     axm1_zoom.set_ylabel("Density", fontsize = 'xx-large')
#     axm1_zoom.set_title("Distribution of m1 for Z = "+metallicity, fontsize = 'xx-large') 
#     axm1_zoom.set_xlim([0,2.5])
    
#     axmtot_zoom.set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 'xx-large')
#     axmtot_zoom.set_ylabel("Density", fontsize = 'xx-large')
#     axmtot_zoom.set_title("Distribution of $m_{tot} = m_1+m_2$ for Z = "+metallicity, fontsize = 'xx-large')
#     axmtot_zoom.set_xlim([0,8])

    axtimes.set_xlabel("delay time [Myr]", fontsize = 'xx-large')
    axtimes.set_ylabel("Density", fontsize = 'xx-large')
    axtimes.set_title("fMT = "+simul_num, fontsize = 'xx-large')
    axtimes.set_xscale('log')
    axtimes.set_yscale('log')
    

    #THIS CODE IS TO AVOID THAT IN THE LEGEND THERE IS A VOID SQUARE AND NOT A LINE (see the notebook I sent you)
    #once I figure out how to sample at the center of each histogram I think I'll drop this out because it is
    #more aesthetic and kawaii
    handles, labels = axm1.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm1.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title = 'Z', title_fontsize = 'xx-large', ncol = 2)
    
    handles, labels = axm2.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm2.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title = 'Z', title_fontsize = 'xx-large', ncol = 2)

    handles, labels = axmtot.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmtot.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title = 'Z', title_fontsize = 'xx-large', ncol = 2)
    
    handles, labels = axq.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axq.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title = 'Z', title_fontsize = 'xx-large', ncol = 2)

    handles, labels = axmchirp.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmchirp.legend(loc = 'center left',handles=temp_handles, labels=labels, fontsize = 'xx-large', title = 'Z', title_fontsize = 'xx-large', ncol = 2)
    
    handles, labels = axtimes.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axtimes.legend(loc = 'lower center', handles=temp_handles, labels=labels, fontsize = 'xx-large', title = 'Z', title_fontsize = 'xx-large', ncol = 2)

#     handles, labels = axm1_zoom.get_legend_handles_labels()   
#     temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
#     axm1_zoom.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

#     handles, labels = axmtot_zoom.get_legend_handles_labels()   
#     temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
#     axmtot_zoom.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    axm1.tick_params( labelsize = 'xx-large' )
    axm2.tick_params( labelsize = 'xx-large' )
    axmtot.tick_params( labelsize = 'xx-large' )
    axq.tick_params( labelsize = 'xx-large' )
    axmchirp.tick_params( labelsize = 'xx-large' )
    axtimes.tick_params( labelsize = 'xx-large' )
#     axm1_zoom.tick_params( labelsize = 'xx-large' )
#     axmtot_zoom.tick_params(labelsize = 'xx-large')


    # Create new directory
    output_dir = "BNS/fixed_fMT/fMT"+sim_numb
    mkdir_p(output_dir)

    fig_m1.savefig('{}/m1.png'.format(output_dir))
    fig_m2.savefig('{}/m2.png'.format(output_dir))
    fig_mtot.savefig('{}/mtot.png'.format(output_dir))
    fig_q.savefig('{}/q.png'.format(output_dir))
    fig_mchirp.savefig('{}/mchirp.png'.format(output_dir))
    fig_times.savefig('{}/delaytimes.png'.format(output_dir))
#     fig_m1_zoom.savefig('{}/m1_zoom.png'.format(output_dir))
#     fig_mtot_zoom.savefig('{}/mtot_zoom.png'.format(output_dir))



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
