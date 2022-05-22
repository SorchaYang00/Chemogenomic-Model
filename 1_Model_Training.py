import pandas as pd
import os
from xgboost import XGBClassifier
import numpy as np
import pickle

    
##### Train a model for each of six descriptor groups #####

os.chdir(r"E:\data\mol_prot\Withlabel")   
files = os.listdir()
  
for file_o in files:
    os.chdir(r"E:\data\mol_prot\Withlabel")
    data = pd.read_csv(file_o)
    data = data.rename(columns = {'0':'index','1':'UniProt'})

    X = np.array(data.iloc[:,5:-1]).astype(np.float64)
    y = np.array(data.iloc[:,-1])
    
    
    model = XGBClassifier(learning_rate=0.26
                           ,max_depth=6
                           ,gamma=1
                           ,n_estimators=700
                           ,subsample=0.6
                           ,colsample_bytree = 0.8
                           ,n_jobs = -1)
    
    model.fit(X,y)              
        
    os.chdir(r'E:\train_model')
    pkl_filename = "{}.pkl".format(file_o.replace('_label.csv',''))
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)      
        
