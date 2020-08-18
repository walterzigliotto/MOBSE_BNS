import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  

#make a color palette of 12 colors
#viridis 
#plasma 
#inferno 
# https://ui.adsabs.harvard.edu/abs/2019PhRvL.123d1102Z/abstract 
# r=d * (m1/m2)**(1./3) 


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
            
def q(dataframe):
    
    m1 = dataframe['m1form[8]']
    m2 = dataframe['m2form[10]']
    
    vector = []
    
    for first, second in zip(m1,m2):
        ratio = second/first
        
        if    ratio <= 1: vector.append(ratio)
        elif  ratio > 1: vector.append(np.power(ratio,-1))
        
    return(np.array(vector))


def plot_assignment1_fixedZ(dataframe):
    
    #make a color palette of 7 colors
    sns.set_palette('inferno', len(fMT))
    
    list_values_met = dataframe.metallicity.unique()
    list_values_fmT = dataframe.fMT.unique()
    
    #initialize space for plots
    fig_m1, axm1         = plt.subplots(1,2,figsize = (30,10), sharey = True)
    fig_m2, axm2         = plt.subplots(1,2,figsize = (30,10), sharey = True)
    fig_mtot, axmtot     = plt.subplots(1,2,figsize = (30,10), sharey = True)
    fig_q, axq           = plt.subplots(1,2,figsize = (30,10), sharey = True)
    
    
    #it returns the metallicities of the list above, but actually code is more flexible
    for fmt in list_values_fmT:

        #retrieve only a single metallicity and the string of fmT value
        subset_data = dataframe[dataframe['fMT'] == fmt]
        simul_num   = fmt
        metallicity = dataframe.iloc[0]['metallicity']
        met_numb = metallicity
        
        #subsubset mergers 
        subset_data_merge = subset_data[subset_data['merge'] == True]
        subset_data_not_merge = subset_data[subset_data['merge'] == False]

        #DATA PREPARATION
        m_tot_merge     = subset_data_merge['m1form[8]']+subset_data_merge['m2form[10]']
        m_tot_not_merge = subset_data_not_merge['m1form[8]']+subset_data_not_merge['m2form[10]']
        
        string_merge = simul_num + '   ({})'.format(len(subset_data_merge))
        string_not_merge = simul_num + '   ({})'.format(len(subset_data_not_merge))

        
        #PLOTTING STUFF for merge
        sns.distplot(subset_data_merge['m1form[8]'], kde = False, label = string_merge, ax = axm1[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(subset_data_merge['m2form[10]'], kde = False, label = string_merge, ax = axm2[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(m_tot_merge, kde = False, label = string_merge, ax = axmtot[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(q(subset_data_merge), kde = False, label = string_merge, ax = axq[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        
        #PLOTTING STUFF for not merge       
        sns.distplot(subset_data_not_merge['m1form[8]'], kde = False, label = string_not_merge, ax = axm1[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(subset_data_not_merge['m2form[10]'], kde = False, label = string_not_merge, ax = axm2[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(m_tot_not_merge, kde = False, label = string_not_merge, ax = axmtot[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(q(subset_data_not_merge), kde = False, label = string_not_merge, ax = axq[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        

    axm1[0].set_xlabel("m1 $[M_\odot]$", fontsize = 'xx-large')
    axm1[0].set_ylabel("Counts", fontsize = 'xx-large')
    axm1[0].set_title("Merge", fontsize = 'xx-large')
    axm1[0].set_yscale('log')
    
    axm2[0].set_xlabel("m2 $[M_\odot]$", fontsize = 'xx-large')
    axm2[0].set_ylabel("Counts", fontsize = 'xx-large')
    axm2[0].set_title("Merge", fontsize = 'xx-large')
    axm2[0].set_yscale('log')
    
    axmtot[0].set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 'xx-large')
    axmtot[0].set_ylabel("Counts", fontsize = 'xx-large')
    axmtot[0].set_title("Merge", fontsize = 'xx-large')
    axmtot[0].set_yscale('log')
    
    axq[0].set_xlabel("q", fontsize = 'xx-large')
    axq[0].set_ylabel("Counts", fontsize = 'xx-large')
    axq[0].set_title("Merge", fontsize = 'xx-large')
    axq[0].set_yscale('log')
    
    axm1[1].set_xlabel("m1 $[M_\odot]$", fontsize = 'xx-large')
    axm1[1].set_ylabel("Counts", fontsize = 'xx-large')
    axm1[1].set_title("Not Merge", fontsize = 'xx-large')
    axm1[1].set_yscale('log')
    
    axm2[1].set_xlabel("m2 $[M_\odot]$", fontsize = 'xx-large')
    axm2[1].set_ylabel("Counts", fontsize = 'xx-large')
    axm2[1].set_title("Not Merge", fontsize = 'xx-large')
    axm2[1].set_yscale('log')
    
    axmtot[1].set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 'xx-large')
    axmtot[1].set_ylabel("Counts", fontsize = 'xx-large')
    axmtot[1].set_title("Not Merge", fontsize = 'xx-large')
    axmtot[1].set_yscale('log')
    
    axq[1].set_xlabel("q", fontsize = 'xx-large')
    axq[1].set_ylabel("Counts", fontsize = 'xx-large')
    axq[1].set_title("Not Merge", fontsize = 'xx-large')
    axq[1].set_yscale('log')
    
    handles, labels = axm1[0].get_legend_handles_labels()
    handles, labelsB = axm1[1].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm1[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'fMT', ncol = 2)
    axm1[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'fMT', ncol = 2)
    
    handles, labels = axm2[0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm2[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'fMT', ncol = 2)
    axm2[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'fMT', ncol = 2)
    
    handles, labels = axmtot[0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmtot[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large',title_fontsize = 'xx-large', title = 'fMT', ncol = 2)
    axmtot[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large',title_fontsize = 'xx-large', title = 'fMT', ncol = 2)

    handles, labels = axq[0].get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axq[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'fMT', ncol = 2)
    axq[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'fMT', ncol = 2)


    axm1[0].tick_params( labelsize = 'xx-large' )
    axm1[0].grid(True, which="both", ls="-",color='0.93')
    axm1[1].tick_params( labelsize = 'xx-large' )
    axm1[1].grid(True, which="both", ls="-",color='0.93')    
    axm2[0].tick_params( labelsize = 'xx-large' )
    axm2[0].grid(True, which="both", ls="-",color='0.93')
    axm2[1].tick_params( labelsize = 'xx-large' )
    axm1[1].grid(True, which="both", ls="-",color='0.93')    
    axmtot[0].tick_params( labelsize = 'xx-large' )
    axmtot[0].grid(True, which="both", ls="-",color='0.93')    
    axmtot[1].tick_params( labelsize = 'xx-large' )
    axmtot[1].grid(True, which="both", ls="-",color='0.93')
    axq[0].tick_params( labelsize = 'xx-large' )
    axq[0].grid(True, which="both", ls="-",color='0.93')    
    axq[1].tick_params( labelsize = 'xx-large' )
    axq[1].grid(True, which="both", ls="-",color='0.93')
    
    # Create new directory
    output_dir = "BNS/mergers/fixed_Z/Z"+met_numb
    mkdir_p(output_dir)
    

    fig_m1.suptitle('m1 distribution - Z = '+met_numb, fontsize = 30)
    fig_m2.suptitle('m2 distribution - Z = '+met_numb, fontsize = 30)
    fig_mtot.suptitle('m tot distribution - Z = '+met_numb, fontsize = 30)
    fig_q.suptitle('q distribution - Z = '+met_numb, fontsize = 30)
    
             
    fig_m1.savefig('{}/m1.png'.format(output_dir))
    fig_m2.savefig('{}/m2.png'.format(output_dir))
    fig_mtot.savefig('{}/mtot.png'.format(output_dir))
    fig_q.savefig('{}/q.png'.format(output_dir))
    
    
#     fig_m1.close()
#     fig_m2.close()
#     fig_mtot.close()
#     fig_q.close()
    
    
def plot_assignment1_fixedfMT(dataframe):
    
    #make a color palette of 12 colors
    sns.set_palette('inferno', len(metallicities))

    list_values_met = dataframe.metallicity.unique()
    list_values_fmT = dataframe.fMT.unique()
    sim_numb = list_values_fmT[0]

    #initialize space for plots
    fig_m1, axm1         = plt.subplots(1,2,figsize = (30,10), sharey = True)
    fig_m2, axm2         = plt.subplots(1,2,figsize = (30,10), sharey = True)
    fig_mtot, axmtot     = plt.subplots(1,2,figsize = (30,10), sharey = True)
    fig_q, axq           = plt.subplots(1,2,figsize = (30,10), sharey = True)

        
    #it returns the metallicities of the list above, but actually code is more flexible
    for metallicity in list_values_met:

        #retrieve only a single metallicity and the string of fmT value
        subset_data = dataframe[dataframe['metallicity'] == metallicity]
        simul_num   = dataframe.iloc[0]['fMT']
        
        #subsubset mergers 
        subset_data_merge = subset_data[subset_data['merge'] == True]
        subset_data_not_merge = subset_data[subset_data['merge'] == False]

        #DATA PREPARATION
        m_tot_merge     = subset_data_merge['m1form[8]']+subset_data_merge['m2form[10]']
        m_tot_not_merge = subset_data_not_merge['m1form[8]']+subset_data_not_merge['m2form[10]']
        
        string_merge = metallicity + '   ({})'.format(len(subset_data_merge))
        string_not_merge = metallicity + '   ({})'.format(len(subset_data_not_merge))
    
         #PLOTTING STUFF for merge
        sns.distplot(subset_data_merge['m1form[8]'], kde = False, label = string_merge, ax = axm1[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(subset_data_merge['m2form[10]'], kde = False, label = string_merge, ax = axm2[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(m_tot_merge, kde = False, label = string_merge, ax = axmtot[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(q(subset_data_merge), kde = False, label = string_merge, ax = axq[0], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        
        #PLOTTING STUFF for not merge       
        sns.distplot(subset_data_not_merge['m1form[8]'], kde = False, label = string_not_merge, ax = axm1[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(subset_data_not_merge['m2form[10]'], kde = False, label = string_not_merge, ax = axm2[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(m_tot_not_merge, kde = False, label = string_not_merge, ax = axmtot[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        sns.distplot(q(subset_data_not_merge), kde = False, label = string_not_merge, ax = axq[1], hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9})
        

    axm1[0].set_xlabel("m1 $[M_\odot]$", fontsize = 'xx-large')
    axm1[0].set_ylabel("Counts", fontsize = 'xx-large')
    axm1[0].set_title("Merge", fontsize = 'xx-large')
    axm1[0].set_yscale('log')

    axm2[0].set_xlabel("m2 $[M_\odot]$", fontsize = 'xx-large')
    axm2[0].set_ylabel("Counts", fontsize = 'xx-large')
    axm2[0].set_title("Merge", fontsize = 'xx-large')
    axm2[0].set_yscale('log')

    axmtot[0].set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 'xx-large')
    axmtot[0].set_ylabel("Counts", fontsize = 'xx-large')
    axmtot[0].set_title("Merge", fontsize = 'xx-large')
    axmtot[0].set_yscale('log')

    axq[0].set_xlabel("q", fontsize = 'xx-large')
    axq[0].set_ylabel("Counts", fontsize = 'xx-large')
    axq[0].set_title("Merge", fontsize = 'xx-large')
    axq[0].set_yscale('log')

    
    axm1[1].set_xlabel("m1 $[M_\odot]$", fontsize = 'xx-large')
    axm1[1].set_ylabel("Counts", fontsize = 'xx-large')
    axm1[1].set_title("Not Merge", fontsize = 'xx-large')
    axm1[1].set_yscale('log')

    axm2[1].set_xlabel("m2 $[M_\odot]$", fontsize = 'xx-large')
    axm2[1].set_ylabel("Counts", fontsize = 'xx-large')
    axm2[1].set_title("Not Merge" , fontsize = 'xx-large')
    axm2[1].set_yscale('log')

    axmtot[1].set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 'xx-large')
    axmtot[1].set_ylabel("Counts", fontsize = 'xx-large')
    axmtot[1].set_title("Not Merge", fontsize = 'xx-large')
    axmtot[1].set_yscale('log')

    axq[1].set_xlabel("q", fontsize = 'xx-large')
    axq[1].set_ylabel("Counts", fontsize = 'xx-large')
    axq[1].set_title("Not Merge", fontsize = 'xx-large')
    axq[1].set_yscale('log')
        
    handles, labels = axm1[0].get_legend_handles_labels()
    handles, labelsB = axm1[1].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm1[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'Z', ncol = 3)
    axm1[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'Z', ncol = 3)
    
    handles, labels = axm2[0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm2[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'Z', ncol = 3)
    axm2[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'Z', ncol = 3)
    
    handles, labels = axmtot[0].get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmtot[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title_fontsize = 'xx-large', title = 'Z', ncol = 3)
    axmtot[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'Z', ncol = 3)

    handles, labels = axq[0].get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axq[0].legend(handles=temp_handles, labels=labels, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'Z', ncol = 3)
    axq[1].legend(handles=temp_handles, labels=labelsB, fontsize = 'xx-large', title_fontsize = 'xx-large',title = 'Z', ncol = 3)


    axm1[0].tick_params( labelsize = 'xx-large' )
    axm1[0].grid(True, which="both", ls="-",color='0.93')
    axm1[1].tick_params( labelsize = 'xx-large' )
    axm1[1].grid(True, which="both", ls="-",color='0.93')    
    axm2[0].tick_params( labelsize = 'xx-large' )
    axm2[0].grid(True, which="both", ls="-",color='0.93')
    axm2[1].tick_params( labelsize = 'xx-large' )
    axm1[1].grid(True, which="both", ls="-",color='0.93')    
    axmtot[0].tick_params( labelsize = 'xx-large' )
    axmtot[0].grid(True, which="both", ls="-",color='0.93')    
    axmtot[1].tick_params( labelsize = 'xx-large' )
    axmtot[1].grid(True, which="both", ls="-",color='0.93')
    axq[0].tick_params( labelsize = 'xx-large' )
    axq[0].grid(True, which="both", ls="-",color='0.93')    
    axq[1].tick_params( labelsize = 'xx-large' )
    axq[1].grid(True, which="both", ls="-",color='0.93')

    # Create new directory
    output_dir = "BNS/mergers/fixedfMT/fMT"+sim_numb
    mkdir_p(output_dir)
    
    fig_m1.suptitle('m1 distribution - fmT = '+simul_num, fontsize = 30)
    fig_m2.suptitle('m2 distribution - fmT = '+simul_num, fontsize = 30)
    fig_mtot.suptitle('m tot distribution - fmT = '+simul_num, fontsize = 30)
    fig_q.suptitle('q distribution - fmT = '+simul_num, fontsize = 30)
    

    fig_m1.savefig('{}/m1.png'.format(output_dir))
    fig_m2.savefig('{}/m2.png'.format(output_dir))
    fig_mtot.savefig('{}/mtot.png'.format(output_dir))
    fig_q.savefig('{}/q.png'.format(output_dir))

#     fig_m1.close()
#     fig_m2.close()
#     fig_mtot.close()
#     fig_q.close()
    
def IDs_from_merge(fMT, metallicity, chunk):
    
    filename = '../modB/simulations_fMT'+fMT+'/A5/'+metallicity+'/chunk'+chunk+'/mergers.out'
    df = pd.read_csv(filename, delim_whitespace = True, header = 0)
    df_modified = df[df.columns[:-1]]
    df_modified.columns = df.columns[1:]
    
    #check over COBs that are neutron stars
    df_NS = df_modified[(df_modified['k1form[7]'] == 13) & (df_modified['k2form[9]'] == 13)]
    BNS_systems_IDs = df_NS.loc[:,'ID[1]'].unique()
    
    return BNS_systems_IDs

for i in fMT:
    temp_dataframesB = []
    for metallicity in metallicities:
        temp_dataframesA = []

        for chunk in chunks:
            
            filename_all = '../modB/simulations_fMT'+i+'/A5/'+metallicity+'/chunk'+chunk+'/COB.out'

            df = pd.read_csv(filename_all, delim_whitespace = True, header = 0)
            df_modified = df[df.columns[:-1]]
            df_modified.columns = df.columns[1:]
            df_subset = df_modified[(df_modified['k1form[7]'] == 13) & (df_modified['k2form[9]'] == 13 ) ]
            df_subset['metallicity'] = metallicity
            df_subset['fMT'] = i
            
            IDs_merge = IDs_from_merge(i, metallicity, chunk)
            merge_flag = df_subset.iloc[:,0].isin(IDs_merge)
            df_subset['merge'] = merge_flag
            
            temp_dataframesA.append(df_subset)

        #the next one is a dataframe containing all a given metallicity
        data_givenZ = pd.concat(temp_dataframesA, ignore_index=True)
        temp_dataframesB.append(data_givenZ)

    data = pd.concat(temp_dataframesB, ignore_index=True)
    plot_assignment1_fixedfMT(data)


for metallicity in metallicities:
    temp_dataframesB = []
    for i in fMT:
        temp_dataframesA = []
        for chunk in chunks:
            filename_all = '../modB/simulations_fMT'+i+'/A5/'+metallicity+'/chunk'+chunk+'/COB.out'

            df = pd.read_csv(filename_all, delim_whitespace = True, header = 0)
            df_modified = df[df.columns[:-1]]
            df_modified.columns = df.columns[1:]
            df_subset = df_modified[(df_modified['k1form[7]'] == 13) & (df_modified['k2form[9]'] == 13 ) ]
            df_subset['metallicity'] = metallicity
            df_subset['fMT'] = i
            
            IDs_merge = IDs_from_merge(i, metallicity, chunk)
            merge_flag = df_subset.iloc[:,0].isin(IDs_merge)
            df_subset['merge'] = merge_flag
            
            temp_dataframesA.append(df_subset)

        #the next one is a dataframe containing all a given metallicity
        data_givenZ = pd.concat(temp_dataframesA, ignore_index=True)
        temp_dataframesB.append(data_givenZ)

    data = pd.concat(temp_dataframesB, ignore_index=True)
    plot_assignment1_fixedZ(data)

    

    