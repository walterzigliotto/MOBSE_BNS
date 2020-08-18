import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   



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

        
def plot_assignment2(dataframe_CE, dataframe_not_CE):
    
    #make a color palette of 12 colors
    sns.set_palette('viridis', 12)

    list_values_met_CE = dataframe_CE.metallicity.unique()
    list_values_fmT_CE = dataframe_CE.fMT.unique()
    sim_numb_CE = list_values_fmT_CE[0]
    met_numb_CE = list_values_met_CE[0]

    list_values_met_not_CE = dataframe_not_CE.metallicity.unique()
    list_values_fmT_not_CE = dataframe_not_CE.fMT.unique()
    sim_numb_not_CE = list_values_fmT_not_CE[0]
    met_numb_not_CE = list_values_met_not_CE[0]

    #initialize space for plots
    fig_mass, axmass  = plt.subplots(2,2,figsize = (30,24), sharey = True)
    fig_mass1, axmass1  = plt.subplots(2,2,figsize = (30,24), sharey = True)

 #   fig_m2, axm2         = plt.subplots(figsize = (18,12))
 #   fig_mtot, axmtot     = plt.subplots(figsize = (18,12))
 #   fig_q, axq           = plt.subplots(figsize = (18,12))
 #  fig_mchirp, axmchirp = plt.subplots(figsize = (18,12))
 # fig_times, axtimes    = plt.subplots(figsize = (18,12))
    


        #PLOTTING STUFF

    #it returns the metallicities of the list above, but actually code is more flexible
    for fmt in list_values_fmT_CE:

        #retrieve only a single metallicity and the string of fmT value
        subset_data_CE = dataframe_CE[dataframe_CE['fMT'] == fmt]
        simul_num_CE   = fmt
        metallicity = dataframe_CE.iloc[0]['metallicity']


        sns.distplot(subset_data_CE['min1[2]'], kde = False, label = "fMT"+simul_num_CE+"  Z="+metallicity, ax = axmass[0,0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)
        sns.distplot(subset_data_CE['min2[3]'], kde = False, label = "fMT"+simul_num_CE+"  Z="+metallicity, ax = axmass[1,0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)

        sns.distplot(subset_data_CE['m1form[8]'], kde = False, label = "fMT"+simul_num_CE+"  Z="+metallicity, ax = axmass1[0,0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)
        sns.distplot(subset_data_CE['m2form[10]'], kde = False, label = "fMT"+simul_num_CE+"  Z="+metallicity, ax = axmass1[1,0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)


    for fmt in list_values_fmT_not_CE:

        #retrieve only a single metallicity and the string of fmT value
        subset_data_not_CE = dataframe_not_CE[dataframe_not_CE['fMT'] == fmt]
        simul_num_not_CE   = fmt
        metallicity = dataframe_not_CE.iloc[0]['metallicity']

        #plot m2
        sns.distplot(subset_data_not_CE['min1[2]'], kde = False, label = "fMT"+simul_num_not_CE+"  Z="+metallicity, ax = axmass[0,1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)
        sns.distplot(subset_data_not_CE['min2[3]'], kde = False, label = "fMT"+simul_num_not_CE+"  Z="+metallicity, ax = axmass[1,1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)

        sns.distplot(subset_data_not_CE['m1form[8]'], kde = False, label = "fMT"+simul_num_not_CE+"  Z="+metallicity, ax = axmass1[0,1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)
        sns.distplot(subset_data_not_CE['m2form[10]'], kde = False, label = "fMT"+simul_num_not_CE+"  Z="+metallicity, ax = axmass1[1,1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist = True)


    axmass[0,0].set_xlabel("min1 $[M_\odot]$", fontsize = 'xx-large')
    axmass[0,0].set_ylabel("Density", fontsize = 'xx-large')
    axmass[0,0].set_title("CE: Distribution of min1 for Z = "+met_numb_CE , fontsize = 'xx-large')

    axmass[0,1].set_xlabel("min1 $[M_\odot]$", fontsize = 'xx-large')
    axmass[0,1].set_ylabel("Density", fontsize = 'xx-large')
    axmass[0,1].set_title("NOT CE: Distribution of min1 for Z = "+met_numb_not_CE, fontsize = 'xx-large')

    axmass[1,0].set_xlabel("min2 $[M_\odot]$", fontsize = 'xx-large')
    axmass[1,0].set_ylabel("Density", fontsize = 'xx-large')
    axmass[1,0].set_title("CE: Distribution of min2 for Z = "+met_numb_CE , fontsize = 'xx-large')

    axmass[1,1].set_xlabel("min2 $[M_\odot]$", fontsize = 'xx-large')
    axmass[1,1].set_ylabel("Density", fontsize = 'xx-large')
    axmass[1,1].set_title("NOT CE: Distribution of min2 for Z = "+met_numb_not_CE, fontsize = 'xx-large')

    axmass1[0,0].set_xlabel("mfin1 $[M_\odot]$", fontsize = 'xx-large')
    axmass1[0,0].set_ylabel("Density", fontsize = 'xx-large')
    axmass1[0,0].set_title("CE: Distribution of mfin1 for Z = "+met_numb_CE , fontsize = 'xx-large')

    axmass1[0,1].set_xlabel("mfin1 $[M_\odot]$", fontsize = 'xx-large')
    axmass1[0,1].set_ylabel("Density", fontsize = 'xx-large')
    axmass1[0,1].set_title("NOT CE: Distribution of mfin1 for Z = "+met_numb_not_CE, fontsize = 'xx-large')

    axmass1[1,0].set_xlabel("mfin2 $[M_\odot]$", fontsize = 'xx-large')
    axmass1[1,0].set_ylabel("Density", fontsize = 'xx-large')
    axmass1[1,0].set_title("CE: Distribution of mfin2 for Z = "+met_numb_CE , fontsize = 'xx-large')

    axmass1[1,1].set_xlabel("mfin2 $[M_\odot]$", fontsize = 'xx-large')
    axmass1[1,1].set_ylabel("Density", fontsize = 'xx-large')
    axmass1[1,1].set_title("NOT CE: Distribution of mfin2 for Z = "+met_numb_not_CE, fontsize = 'xx-large')
    


    #THIS CODE IS TO AVOID THAT IN THE LEGEND THERE IS A VOID SQUARE AND NOT A LINE (see the notebook I sent you)
    #once I figure out how to sample at the center of each histogram I think I'll drop this out because it is
    #more aesthetic and kawaii
    handles, labels = axmass[0,0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass[0,0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    handles, labels = axmass[0,1].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass[0,1].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    handles, labels = axmass[1,0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass[1,0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    handles, labels = axmass[1,1].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass[1,1].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    handles, labels = axmass1[0,0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass1[0,0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    handles, labels = axmass1[0,1].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass1[0,1].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

    handles, labels = axmass1[1,0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass1[1,0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    handles, labels = axmass1[1,1].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmass1[1,1].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')

#     handles, labels = axmtot_zoom.get_legend_handles_labels()   
#     temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
#     axmtot_zoom.legend(handles=temp_handles, labels=labels, fontsize = 'xx-large')
    
    axmass[0,0].tick_params( labelsize = 'xx-large' )
    axmass[0,1].tick_params( labelsize = 'xx-large' )
    axmass[1,0].tick_params( labelsize = 'xx-large' )
    axmass[1,1].tick_params( labelsize = 'xx-large' )

    axmass1[0,0].tick_params( labelsize = 'xx-large' )
    axmass1[0,1].tick_params( labelsize = 'xx-large' )
    axmass1[1,0].tick_params( labelsize = 'xx-large' )
    axmass1[1,1].tick_params( labelsize = 'xx-large' )

    
    fig_mass.show()
    fig_mass1.show()
#    fig_mtot.show()
#    fig_q.show()
#    fig_mchirp.show()
#    fig_times.show()
#     fig_m1_zoom.show()
#     fig_mtot_zoom.show()


    # Create new directory
    output_dir = "CE_NOTCE_BNS/fixed_Z/Z"+met_numb_CE
    mkdir_p(output_dir)

    fig_mass.savefig('{}/massin.png'.format(output_dir))
    fig_mass1.savefig('{}/massfin.png'.format(output_dir))


for metallicity in metallicities:
    temp_dataframesBCE = []
    temp_dataframesBnotCE = []

    for i in fMT:
        temp_dataframesACE = []
        temp_dataframesnotCE = []

        for chunk in chunks:
            filename = '../modB/simulations_fMT'+i+'/A5/'+metallicity+'/chunk'+chunk+'/COB.out'

            df = pd.read_csv(filename, delim_whitespace = True, header = 0)
            df_modified = df[df.columns[:-1]]
            df_modified.columns = df.columns[1:]

            #we want to extract NSBH systems
            df_NS = df_modified[(df_modified['k1form[7]'] == 13) & (df_modified['k2form[9]'] == 13)]
               
            df_NS['metallicity'] = metallicity
            df_NS['fMT'] = i

            df_NS_CE = df_NS[df_NS['label[18]'] == 'COMENV']
            df_NS_not_CE = df_NS[df_NS['label[18]'] != 'COMENV']

            temp_dataframesACE.append(df_NS_CE)
            temp_dataframesnotCE.append(df_NS_not_CE)

        #the next one is a dataframe containing all a given metallicity
        data_givenZCE = pd.concat(temp_dataframesACE, ignore_index=True)
        data_givenZnotCE = pd.concat(temp_dataframesnotCE, ignore_index=True)

        temp_dataframesBCE.append(data_givenZCE)
        temp_dataframesBnotCE.append(data_givenZnotCE)

    dataCE = pd.concat(temp_dataframesBCE, ignore_index=True)
    datanotCE = pd.concat(temp_dataframesBnotCE, ignore_index=True)
    plot_assignment2(dataCE, datanotCE)
