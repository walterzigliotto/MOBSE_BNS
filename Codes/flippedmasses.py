import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import bokeh.palettes as palette # For palette of colors

class plot_hist_fMT:
 
    
    def flip_fraction(self,df,fMT,metal):
        # Plot fraction of flipped mass as function of fMT

        num = []
        den = []
       
        for i in range(0,len(fMT)):
            a = 0
            b = 0
            for j in range(0,len(metal)):
                    a += sum(df[i][j]['flip_mask'])
                    b += df[i][j]['flip_mask'].shape[0]
            num.append(a)
            den.append(b)
                
        x_fMT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7,1.0]
        frac = np.array(num)/np.array(den)

        fig, ax = plt.subplots(figsize=(9,7))
        plt.plot(x_fMT,frac,'o',color='blue')
        ax.set_title('Fraction of masses of the final COs that flip ',fontsize=20)
        ax.set_xlabel('fMT',fontsize=18)
        ax.set_ylabel('Fraction',fontsize=20)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_axisbelow(True)           
        
        plt.savefig('masses_1assgn/flipmasses.png', format='png')
        plt.close()



plot_hist_fMT = plot_hist_fMT()
fMT = ['01','02','03','04','05','07','1']
metal = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']

chunk = ['0','1','2','3','4']

df_fMT = []


for i in fMT:

    df_metal = []

    for j in metal:

        df_chunk = []

        for m in chunk:
            path = '/data1/collaborator/modB/simulations_fMT'+i+'/A5/'+j+'/chunk'+m
        
            df = pd.read_csv(path+"/mergers.out",delim_whitespace = True, header = 0)
            df = df.apply(pd.to_numeric, errors='ignore')
            df_modified = df[df.columns[:-1]]
            df_modified.columns = df.columns[1:]
          
            df_subset = df_modified[(df_modified['k1form[7]'] == 13) & (df_modified['k2form[9]'] == 13 ) ]
            # Ordering by m1>m2
            mask = df_subset['m2form[10]']>df_subset['m1form[8]']
            df_subset['flip_mask'] = mask
           
            df_chunk.append(df_subset)

        df_metal.append(pd.concat(df_chunk, ignore_index=True))

    # Concatenation of dataframes with metallicity 0.012, 0.016, 0.02
    df_metal_u = pd.concat(df_metal[-3:],ignore_index=True)
    df_metal = df_metal[:-3]
    
    df_metal.append(df_metal_u)

    df_fMT.append(df_metal)

#######################

metallicity = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012-0.016-0.02']

plot_hist_fMT.flip_fraction(df_fMT,fMT,metallicity)



