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

chunks = ['0', '1', '2', '3', '4']
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

    
def import_database_ID_nCEs():
    
    filename = 'BNS/assignment02/IDs_and_CEs_flags.csv'
    df = pd.read_csv(filename, header = 0, dtype = {'fMT': object, 'Z' : object, 'chunk': object})

    return df


def import_database(filename):
    
    path = 'BNS/assignment02/'
    df = pd.read_csv(path+filename, header = 0, dtype = {'fMT': object, 'Z' : object, 'chunk': object})

    return df


def interoccurrence_plot(df):
    
    fig, ax = plt.subplots(1,1)
    
    ax.hist(df.iloc[:,0], density = True, label = 'data', histtype = 'step', color = 'firebrick')
    ax.set_yscale('log')
    ax.set_xlabel('Time [Myear]')
    ax.set_ylabel('density')
    ax.set_title("Time distribution")
    
    handles, labels = ax.get_legend_handles_labels()
    temp_handles = [Line2D([], [], c=h.get_edgecolor(), linewidth = 1) for h in handles]
    
    plt.legend(handles=temp_handles, labels=labels)


    
    plt.savefig("BNS/assignment02/2A_2B_time_difference.png")


times   = import_database('diff_times_CE.csv')


                                                             
start_time = time.time()
interoccurrence_plot(times)
final_time   = time.time()
print("Done with plot :", final_time - start_time, "\n" )