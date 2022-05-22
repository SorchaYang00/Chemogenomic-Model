iport pandas as pd
import numpy as np
import os
import pickle



def Getconc(mol, pro):
    
    dic = {'mol2d':mol2d, 'ecfp4':ecfp4, 
           'maccs':maccs, 'proa':proa, 'prob':prob}
    
    mol = dic[mol].copy()
    mol = mol.iloc[:,3:]   ###### drop the unrelated info which are not descriptors #####
    mol = np.array(mol)
    mol_859 = np.repeat(mol,859,axis=0)
    mol_859 = pd.DataFrame(mol_859)
    
    pro = dic[pro]
    pro_0 = pro.copy()
    pro_0.drop('index',axis=1,inplace=True)  ###### drop the unrelated info which are not descriptors #####
    pro = np.array(pro_0)
    pro_n = np.tile(pro_0,(num,1))
    pro_n = pd.DataFrame(pro_n)
    mol_pro = pd.concat([mol_859,pro_n],axis =1,ignore_index =True, sort =False)
    nump = np.array(mol_pro)
    
    return nump


def Getmodel(modelname):
    os.chdir(r"E:\train_model")

    pkl_filename = modelname
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)     
    return model



if  "__main__" == __name__:
    

    mols= pd.read_csv(r"E:\mols_for_prediction.csv")
        
    ecfp4 = pd.read_csv(r"E:\mols_ECFP4.csv")    
    maccs = pd.read_csv(r"E:\mols_MACCS.csv")
    mol2d = pd.read_csv(r"E:\mols_Mol2d.csv")

    proa = pd.read_csv(r"E:\data\ProA.csv")
    prob = pd.read_csv(r"E:\data\ProB.csv")


       
#### =靶标的重复数量 ######
    num = len(mols)

        
    X1 = Getconc('ecfp4','proa')
    X2 = Getconc('ecfp4','prob')
    X3 = Getconc('maccs','proa')
    X4 = Getconc('maccs','prob')
    X5 = Getconc('mol2d', 'proa')
    X6 = Getconc('mol2d', 'prob')
    
    model_1 = Getmodel('ECFP4_proa.pkl')
    model_2 = Getmodel('ECFP4_prob.pkl')
    model_3 = Getmodel('MACCS_proa.pkl')
    model_4 = Getmodel('MACCS_prob.pkl')
    model_5 = Getmodel('Mol2d_proa.pkl')
    model_6 = Getmodel('Mol2d_prob.pkl')


    y_proba_1 = model_1.predict_proba(X1)[:,1]
    y_proba_2 = model_2.predict_proba(X2)[:,1]
    y_proba_3 = model_3.predict_proba(X3)[:,1]
    y_proba_4 = model_4.predict_proba(X4)[:,1]
    y_proba_5 = model_5.predict_proba(X5)[:,1]
    y_proba_6 = model_6.predict_proba(X6)[:,1]
    
    ###### prediction probabilities for individual models #######
    y_proba_stack0 = np.vstack((y_proba_1, y_proba_2, y_proba_3, y_proba_4, y_proba_5, y_proba_6)).T
    df_stack = pd.DataFrame(y_proba_stack0)
    
    ###### calculate ensemble probabilities ######
    df_stack['average'] = df_stack.apply(lambda x: np.average(x), axis=1
                                         

    
