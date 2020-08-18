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
fMT_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.7', '1']
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
    
    #make a color palette of 7 colors
    sns.set_palette('inferno', 12)
    

    list_values_met = dataframe.metallicity.unique()
    list_values_fmT = dataframe.fMT.unique()
    sim_numb = list_values_fmT[0]
    

    #initialize space for plots
    fig_m1, axm1         = plt.subplots(figsize = (9,7))
    fig_m2, axm2         = plt.subplots(figsize = (9,7))
    fig_mtot, axmtot     = plt.subplots(figsize = (9,7))
    fig_q, axq           = plt.subplots(figsize = (9,7))
    fig_mchirp, axmchirp = plt.subplots(figsize = (9,7))
    fig_diff1, axdiff1    = plt.subplots(figsize = (9,7))
    fig_diff2, axdiff2    = plt.subplots(figsize = (9,7))
    fig_m1_zoom  , axm1_zoom = plt.subplots(figsize = (9,7))
    fig_m2_zoom, axm2_zoom = plt.subplots(figsize = (9,7))
    fig_mtot_zoom, axmtot_zoom = plt.subplots(figsize = (9,7))
    fig_q_zoom, axq_zoom = plt.subplots(figsize = (9,7))
    fig_mchirp_zoom, axmchirp_zoom = plt.subplots(figsize = (9,7))
    fig_zams_1, axzams_1   = plt.subplots(figsize = (9,7))
    fig_zams_2, axzams_2    = plt.subplots(figsize = (9,7))  
    
   #it returns the metallicities of the list above, but actually code is more flexible
    for metallicity in list_values_met:
        

        #retrieve only a single metallicity and the string of fmT value
        subset_data = dataframe[dataframe['metallicity'] == metallicity]
        simul_num   = dataframe.iloc[0]['fMT']
      
        #DATA PREPARATION
        m_tot   = subset_data['m1form[8]']+subset_data['m2form[10]']
        q       = subset_data['m2form[10]']/subset_data['m1form[8]']
        m_chirp = (subset_data['m2form[10]']*subset_data['m1form[8]'])**(3/5.)/(subset_data['m1form[8]']+subset_data['m2form[10]'])**(1/5.)
        diff1 = subset_data['min1[2]']-subset_data['m1form[8]']
        diff2 = subset_data['min2[3]']-subset_data['m2form[10]']
        zams_1   = subset_data['min1[2]']
        zams_2   = subset_data['min2[3]']
        


        #PLOTTING STUFF
        #first m1 plot, ax=axm1 is to superimpose to an unique plot the all different metallicities
        sns.distplot(subset_data['m1form[8]'], kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axm1, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        sns.distplot(subset_data['m1form[8]'], kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axm1_zoom, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot m2
        sns.distplot(subset_data['m2form[10]'], kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axm2, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        sns.distplot(subset_data['m2form[10]'], kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axm2_zoom, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot mtot
        sns.distplot(m_tot, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axmtot, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        sns.distplot(m_tot, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axmtot_zoom, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)


        #plot q
        sns.distplot(q, kde = False, label = metallicity, ax = axq, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        sns.distplot(q, kde = False, label = metallicity, ax = axq_zoom, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        #plot m_chirp
        sns.distplot(m_chirp, kde = False, label = "  Z="+metallicity, ax = axmchirp, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        sns.distplot(m_chirp, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axmchirp_zoom, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)


        #plot masses differences
        
        sns.distplot(diff1, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axdiff1, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        sns.distplot(diff2, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axdiff2, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)

        sns.distplot(diff1, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axzams_1, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        sns.distplot(diff2, kde = False, label = "fMT"+simul_num+"  Z="+metallicity, ax = axzams_2, hist_kws={"histtype": "step", "linewidth" : 2, "alpha" : 0.9}, norm_hist=True)
        

       

    axm1.set_xlabel("m1 $[M_\odot]$", fontsize = 20)
    axm1.set_ylabel("Density", fontsize = 20)
    axm1.set_title("Distribution of m1 for fMT = "+sim_numb, fontsize = 20)

    axm2.set_xlabel("m2 $[M_\odot]$", fontsize = 20)
    axm2.set_ylabel("Density", fontsize = 20)
    axm2.set_title("Distribution of m2 for fMT = "+sim_numb, fontsize = 20)

    axmtot.set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 20)
    axmtot.set_ylabel("Density", fontsize = 20)
    axmtot.set_title("Distribution of $m_{tot} = m_1+m_2$ for fMT = "+sim_numb, fontsize = 20)

    axq.set_xlabel("q", fontsize = 20)
    axq.set_ylabel("Density", fontsize = 20)
    axq.set_title("Distribution of $q= m_2 / m_1$ for fMT = "+sim_numb  , fontsize = 20)

    axmchirp.set_xlabel("$m_{chirp}$ $[M_\odot]$", fontsize = 20)
    axmchirp.set_ylabel("Density", fontsize = 20)
    axmchirp.set_title("Distribution of $m_{chirp}$ for fMT = "+sim_numb , fontsize = 20)

    axdiff1.set_xlabel("$\Delta$m1", fontsize = 20)
    axdiff1.set_ylabel("Density", fontsize = 20)
    axdiff1.set_title("Distribution of $\Delta$m1 for fMT = "+sim_numb , fontsize = 20)

    axdiff2.set_xlabel("$\Delta$m2", fontsize = 20)
    axdiff2.set_ylabel("Density", fontsize = 20)
    axdiff2.set_title("Distribution of $\Delta$m2 for fMT = "+sim_numb , fontsize = 20)

    axzams_1.set_xlabel("ZAMS 1", fontsize = 20)
    axzams_1.set_ylabel("Density", fontsize = 20)
    axzams_1.set_title("Distribution of ZAMS 1 for fMT = "+sim_numb, fontsize = 20)

    axzams_2.set_xlabel("ZAMS 2", fontsize = 20)
    axzams_2.set_ylabel("Density", fontsize = 20)
    axzams_2.set_title("Distribution of ZAMS 2 for fMT = "+sim_numb , fontsize = 20)

    axm1_zoom.set_xlabel("m1 $[M_\odot]$", fontsize = 20)
    axm1_zoom.set_ylabel("Density", fontsize = 20)
    axm1_zoom.set_title("Distribution of m1 for fMT = "+sim_numb , fontsize = 20) 
    axm1_zoom.set_xlim([1,1.75])

    axm2_zoom.set_xlabel("m2 $[M_\odot]$", fontsize = 20)
    axm2_zoom.set_ylabel("Density", fontsize = 20)
    axm2_zoom.set_title("Distribution of m2 for fMT = "+sim_numb, fontsize = 20) 
    axm2_zoom.set_xlim([1,1.7])
    
    axmtot_zoom.set_xlabel("$m_{tot}$ $[M_\odot]$", fontsize = 20)
    axmtot_zoom.set_ylabel("Density", fontsize = 20)
    axmtot_zoom.set_title("Distribution of $m_{tot} = m_1+m_2$ for fMT = "+sim_numb, fontsize = 20)
    axmtot_zoom.set_xlim([2.25,4])

    axmchirp_zoom.set_xlabel("$m_{chirp}$ $[M_\odot]$ for fMT = "+sim_numb, fontsize = 20)
    axmchirp_zoom.set_ylabel("Density", fontsize = 20)
    axmchirp_zoom.set_title("Distribution of $m_{chirp}$ for fMT = "+sim_numb, fontsize = 20)
    axmchirp_zoom.set_xlim([1,1.6])

    axq_zoom.set_xlabel("q", fontsize = 20)
    axq_zoom.set_ylabel("Density", fontsize = 20)
    axq_zoom.set_title("Distribution of $q= m_2 / m_1$ for fMT = "+sim_numb, fontsize = 20)
    axq_zoom.set_xlim([0.8,1.2])


    #THIS CODE IS TO AVOID THAT IN THE LEGEND THERE IS A VOID SQUARE AND NOT A LINE 
    handles, labels = axm1.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm1.legend(handles=temp_handles, labels=labels, fontsize = 16)
    
    handles, labels = axm2.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axm2.legend(handles=temp_handles, labels=labels, fontsize = 16)
    
    handles, labels = axmtot.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmtot.legend(handles=temp_handles, labels=labels, fontsize = 16)
    
    handles, labels = axq.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axq.legend(loc = 'center left',handles=temp_handles, labels=labels, fontsize = 16,
    title = 'Z', title_fontsize = 16)

    handles, labels = axmchirp.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axmchirp.legend(loc = 'center right', handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axdiff1.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axdiff1.legend(loc = 'center right', handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axdiff2.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axdiff2.legend(loc = 'center right', handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axzams_1.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axzams_1.legend(loc = 'center right', handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axzams_2.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 2) for h in handles]
    axzams_2.legend(loc = 'center right', handles=temp_handles, labels=labels, fontsize = 16) 
 
    handles, labels = axm1_zoom.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    axm1_zoom.legend(handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axm2_zoom.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    axm2_zoom.legend(handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axmtot_zoom.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles] 
    axmtot_zoom.legend(handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axmchirp_zoom.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles] 
    axmchirp_zoom.legend(handles=temp_handles, labels=labels, fontsize = 16)

    handles, labels = axq_zoom.get_legend_handles_labels()   
    temp_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles] 
    axq_zoom.legend(loc = 'center left',handles=temp_handles, labels=labels, fontsize = 16,title = 'Z', title_fontsize = 16)
     
    


    
    axm1.tick_params( labelsize = 20 )
    axm2.tick_params( labelsize = 20 )
    axmtot.tick_params( labelsize = 20 )
    axq.tick_params( labelsize = 20 )
    axmchirp.tick_params( labelsize = 20 )
    axdiff1.tick_params( labelsize = 20 )
    axdiff2.tick_params( labelsize = 20 )
    axzams_1.tick_params( labelsize = 20 )
    axzams_2.tick_params( labelsize = 20 )
    axm1_zoom.tick_params( labelsize = 20 )
    axm2_zoom.tick_params( labelsize = 20 )
    axmtot_zoom.tick_params(labelsize = 20)
    axmchirp_zoom.tick_params( labelsize = 20 )
    axq_zoom.tick_params( labelsize = 20 )

    

    # Create new directory
    output_dir = "masses_1assgn/fixed_fMT/fMT"+sim_numb
    mkdir_p(output_dir)

    fig_m1.savefig('{}/m1.png'.format(output_dir))
    fig_m2.savefig('{}/m2.png'.format(output_dir))
    fig_mtot.savefig('{}/mtot.png'.format(output_dir))
    fig_q.savefig('{}/q.png'.format(output_dir))
    fig_mchirp.savefig('{}/mchirp.png'.format(output_dir))
    fig_diff1.savefig('{}/diff1.png'.format(output_dir))
    fig_diff2.savefig('{}/diff2.png'.format(output_dir))
    fig_zams_1.savefig('{}/zams1.png'.format(output_dir))
    fig_zams_2.savefig('{}/zams2.png'.format(output_dir))
    fig_m1_zoom.savefig('{}/m1_zoom.png'.format(output_dir))
    fig_mtot_zoom.savefig('{}/mtot_zoom.png'.format(output_dir))
    fig_m2_zoom.savefig('{}/m2_zoom.png'.format(output_dir))
    fig_mchirp_zoom.savefig('{}/mchirp_zoom.png'.format(output_dir))
    fig_q_zoom.savefig('{}/q_zoom.png'.format(output_dir))




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
