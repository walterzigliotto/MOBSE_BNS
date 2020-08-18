import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time



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



def import_data():
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
    final_time = time.time()
    print("Total time execution: ", final_time - start_time )
    
    return datatotal


def a_coefficient(dataframe):
    fMT_csv   = ['0.1','0.2','0.3','0.4','0.5','0.7','1']
    col_names =  ['a', 'err_a' , 'Z', 'fMT']
    results   = pd.DataFrame(columns = col_names)
    
    #subset for fMT
    for sim_numb, fMT_to_csv in zip( fMT, fMT_csv ) :
        subset_dataframe = dataframe[dataframe['fMT'] == sim_numb]
        for Z in metallicities:
            start_time = time.time()
            
            subsubset_dataframe = subset_dataframe[subset_dataframe['metallicity'] == Z]
            
            subsubset_dataframe = subsubset_dataframe[(subsubset_dataframe['tmerg[11]'] > 1e2) & (subsubset_dataframe['tmerg[11]'] < 1e4)]
            
            #now set the log binning set on the min and the max 
            min_time, max_time = subsubset_dataframe['tmerg[11]'].min(), subsubset_dataframe['tmerg[11]'].max()
            #set bins to 50 (is there a better way to calculate it?)
            log_bins_interp = np.logspace(np.log10(min_time),np.log10(max_time),50)
            
            histog, edges = np.histogram( subsubset_dataframe['tmerg[11]'], bins=log_bins_interp , density=True)
            
            diff = (np.log(edges[1]) - np.log(edges[0]))
            x = [ (diff*(i+1)+0.5*diff) for i in range(len(edges)-1)]
            y = np.log(histog)

            model, V = np.polyfit(x, y, 1, cov=True)
                  
            fit_result = {'a': model[0], 'err_a': np.sqrt(V[0][0]) , 'Z' : Z, 'fMT' : fMT_to_csv  }
            
            results = results.append(fit_result, ignore_index = True)
            final_time = time.time()
            print("Finished for fMT", sim_numb , "  Z", Z, "    with time:", final_time - start_time )
    return results
            
            
            
            
            
data = import_data()
result = a_coefficient(data)

result.to_csv('powerlawfit.csv', index=False)
            
       
            


