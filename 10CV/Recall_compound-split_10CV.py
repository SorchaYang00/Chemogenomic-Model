import time
import datetime
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,Dropout,Average,Multiply,Concatenate
from sklearn.metrics import confusion_matrix,accuracy_score,auc,roc_auc_score
from sklearn.metrics import precision_score,recall_score
import pandas as pd
import os
from xgboost import XGBClassifier
import numpy as np
import pickle


#num = 1000

def Getconc(mol, pro, x, y):
    struc = mols.iloc[x:y,:]

    mol = dic[mol].copy()
    mol = mol[mol['index'].isin(struc['index'])]
    mol = mol.sort_values(['index'])
    mol = mol.iloc[:,3:]
    mol = np.array(mol)
    mol_859 = np.repeat(mol,859,axis=0)
    mol_859 = pd.DataFrame(mol_859)
    
    pro = dic[pro]
    pro_0 = pro.copy()
    pro_0.drop('index',axis=1,inplace=True)
    pro = np.array(pro_0)
    pro_n = np.tile(pro_0,(num,1))
    pro_n = pd.DataFrame(pro_n)
    mol_pro = pd.concat([mol_859,pro_n],axis =1,ignore_index =True, sort =False)
    nump = np.array(mol_pro)
    
    return nump

def Constract(all_relation, relation):
    all_relation = all_relation.set_index(['index','UniProt'])
    relation = relation.set_index(['index','UniProt'])
    train = all_relation[~all_relation.index.isin(relation.index)]
    train = train.sort_values(['index','UniProt'])
    train = train.reset_index(drop=False)
    return train

def Getindex(relation,all_relation):
    all_relation = all_relation.set_index(['index','UniProt'])
    relation = relation.set_index(['index','UniProt'])
    correct = relation[relation.index.isin(all_relation.index)]
    correct = correct.reset_index(drop=False)
    return correct  


def Getmodel(modelname):
    os.chdir(r"E:\student\ysq\PCM\Nova\drug-split\train_model")
    model = {'X1':'{}_ECFP4_proa_{}.pkl'.format(random_state, i)
            , 'X2':'{}_ECFP4_prob_{}.pkl'.format(random_state, i)
            ,'X3': '{}_MACCS_proa_{}.pkl'.format(random_state, i)
            , 'X4':'{}_MACCS_prob_{}.pkl'.format(random_state,i)
            ,'X5': '{}_Mol2d_proa_{}.pkl'.format(random_state,i)
            , 'X6':'{}_Mol2d_prob_{}.pkl'.format(random_state,i)}

    pkl_filename = model[modelname]
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)     
    return model


if  "__main__" == __name__:
    
    
    mol2d = pd.read_csv(r"E:\student\ysq\PCM\data\data\bindingdb_mol_2d.csv")
    maccs = pd.read_csv(r"E:\student\ysq\PCM\data\data\bindingdb_mol_maccs.csv")
    ecfp4 = pd.read_csv(r"E:\student\ysq\PCM\data\data\bindingdb_mol_ecfp4.csv")
    proa = pd.read_csv(r"E:\student\ysq\PCM\data\data\proa.csv")
    prob = pd.read_csv(r"E:\student\ysq\PCM\data\data\prob.csv")
    all_relation = pd.read_csv(r"E:\student\ysq\PCM\data\next_data\relation_label.csv")
    
    ###20200913, 20200914, 20200915,20200916,20200917,20200918, 20200919,20200920,20200921, 20200922
    random_states = [20200913, 20200914, 20200915,20200916,20200917,20200918, 20200919,20200920,20200921, 20200922]
    for random_state in random_states: 
        middle_res = pd.DataFrame()
        res = pd.DataFrame()
        
        for i in range(1,10):
            relation_0 = pd.read_csv(r"E:\student\ysq\PCM\Nova\drug-split\10CV\{}_10fold.csv".format(random_state))
            relation = relation_0[relation_0['fold_ECFP4_proa']==i]
            train = Constract(all_relation, relation)
            relation = relation[relation['label_ECFP4_proa']==1]
            mols = relation.drop_duplicates(['index'])
            mols= mols.sort_values(['index'])
            
        
        
            dic = {'mol2d':mol2d, 'ecfp4':ecfp4, 
                   'maccs':maccs, 'proa':proa, 'prob':prob}
            
            
            model = {'X1':'{}_ECFP4_proa_{}.pkl'.format(random_state, i)
                    , 'X2':'{}_ECFP4_prob_{}.pkl'.format(random_state, i)
                    ,'X3': '{}_MACCS_proa_{}.pkl'.format(random_state, i)
                    , 'X4':'{}_MACCS_prob_{}.pkl'.format(random_state,i)
                    ,'X5': '{}_Mol2d_proa_{}.pkl'.format(random_state,i)
                    , 'X6':'{}_Mol2d_prob_{}.pkl'.format(random_state,i)}
            
            
        #    idx1 = range(0,100,10)
        #    idx2 = range(10,110,10)
#            idx1 = range(0,len(mols)-1000,1000)
#            idx2 = range(1000,len(mols),1000)
            idx1 = [(len(mols)//1000)*1000]
            idx2 = [len(mols)]
            
            
        #### 注意修改靶标的重复数量 ######
#            num = 1000 
            num = len(mols)%1000


            
            

            pro_name = np.array(prob.iloc[:,0:1])
            for x,y in zip(idx1,idx2):
                
                X1 = Getconc('ecfp4','proa', x, y)
                X2 = Getconc('ecfp4','prob', x, y)
                X3 = Getconc('maccs','proa', x, y)
                X4 = Getconc('maccs','prob', x, y)
                X5 = Getconc('mol2d', 'proa', x, y)
                X6 = Getconc('mol2d', 'prob', x, y)
                
                model_1 = Getmodel('X1')
                model_2 = Getmodel('X2')
                model_3 = Getmodel('X3')
                model_4 = Getmodel('X4')
                model_5 = Getmodel('X5')
                model_6 = Getmodel('X6')
        #        model_loge = Getmodel('loge')
                
                y_proba_1 = model_1.predict_proba(X1)[:,1]
                y_proba_2 = model_2.predict_proba(X2)[:,1]
                y_proba_3 = model_3.predict_proba(X3)[:,1]
                y_proba_4 = model_4.predict_proba(X4)[:,1]
                y_proba_5 = model_5.predict_proba(X5)[:,1]
                y_proba_6 = model_6.predict_proba(X6)[:,1]
                
                
                y_proba_stack0 = np.vstack((y_proba_1, y_proba_2, y_proba_3, y_proba_4, y_proba_5, y_proba_6)).T
        #        y_proba_stack1 = model_loge.predict_proba(y_proba_stack0)[:,1]
                
                df_stack = pd.DataFrame(y_proba_stack0, columns = list(model.keys())[0:])
                df_stack['average'] = df_stack.apply(lambda x: np.average(x), axis=1)
        #        df_stack['maxism'] = df_stack.iloc[:,:-1].apply(lambda x: max(x), axis=1)
        #        df_stack['stacked'] =  y_proba_stack1
                
                out = mols.iloc[x:y,0:1]
                out = np.array(out.sort_values(['index']))
                out = np.repeat(out,859,axis=0)
                out = pd.DataFrame(out)
                out.columns = ['index']        
                tittle_pro = np.tile(pro_name,(num,1))
                out['UniProt'] = tittle_pro
                out = pd.concat([out,df_stack],axis=1)
                
        #        methods = ['X1', 'X2','X3','X4','X5','X6', 'average','maxism','stacked']
                methods = ['X1', 'X2','X3','X4','X5','X6', 'average']
                
                for item in methods:
                    out['rank_1_{}'.format(item)] = out[item].groupby(out['index']).rank(ascending=False,method = 'min')
                    
                out = Constract(out, train)
                
                for item in methods:
                    out['rank_2_{}'.format(item)] = out[item].groupby(out['index']).rank(ascending=False,method = 'min')
                
            
                middle_res = pd.concat([middle_res,out])
               
                out = Getindex(out, relation)
                res = pd.concat([res,out])
                res.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\recall\watch_recall_test.csv",index=False)
                
#            middle_res.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\recall\{}_recall_test.gzip".format(random_state),index=False)
#            res.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\recall\{}_recall_test.csv".format(random_state),index=False)
#
            middle_res.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\recall\{}_recall_test.gzip".format(random_state),index=False
                                  ,compression='gzip', mode='a', header=False)
            res.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\recall\{}_recall_test.csv".format(random_state),index=False
                                  , mode='a', header=False)
