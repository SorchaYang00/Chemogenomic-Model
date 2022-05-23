import time
import datetime
from sklearn.model_selection import GroupKFold,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,auc,roc_auc_score
from sklearn.metrics import precision_score,recall_score
import pandas as pd
import os
from xgboost import XGBClassifier
import numpy as np
import pickle
from sklearn.utils import shuffle


def GetMetrics(y_test, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).flatten()
    acc = '%.3f'%accuracy_score(y_test,y_pred)
    AUC = '%.3f'%roc_auc_score(y_test,y_pred_proba)
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    df = pd.DataFrame({'Accuracy':[float(acc)],
                       'Sensitivity':[float(sens)],
                       'Specificity':[float(spec)],
                       'AUC':[float(AUC)]
                       })
    return df



def Constract(all_relation, relation):
    all_relation = all_relation.set_index(['index','UniProt'])
    relation = relation.set_index(['index','UniProt'])
    train = all_relation[~all_relation.index.isin(relation.index)]
    train = train.reset_index(drop=False)
    train = train.sort_values(['index','UniProt'])
    train = train.drop_duplicates()
    return train


    
   
if '__main__' == __name__:
   
#    # 范围时间
#    string = '2020-09-28 04:40:40'
#    time1 = datetime.datetime.strptime(string,'%Y-%m-%d %H:%M:%S')
#    # 当前时间
#   
#     
#    # 判断当前时间是否在范围时间内
#    while True:
#        n_time = datetime.datetime.now()
#        if n_time > time1:
#            break
#        else:
#            time.sleep(1200)


    relation = pd.read_csv(r"E:\student\ysq\PCM\data\next_data\relation_label.csv")


    groups = np.array(relation['index'])
    
    os.chdir(r"E:\student\ysq\PCM\data\mol_prot\Withlabel")
    files = os.listdir()

    random_states = [20200913, 20200914, 20200915,20200916,20200917,20200918,20200919,20200920,20200921, 20200922]

    for random_state in random_states:

        result_final = pd.DataFrame()
        result_sta = pd.DataFrame()    
           
        for file_o in files:
            os.chdir(r"E:\student\ysq\PCM\data\mol_prot\Withlabel")
            data = pd.read_csv(file_o)
            data = data.rename(columns = {'0':'index','1':'UniProt'})

    
            X = np.array(data.iloc[:,:-1])
            y = np.array(data.iloc[:,-1])
            
            skf = GroupKFold(n_splits=10)
            
            
            result = pd.DataFrame()
            fold = 0
            X_s, y_s, groups_s = shuffle(X,y, groups, random_state=random_state)
            
            for train_index, test_index in skf.split(X_s, y_s, groups_s):
                fold += 1
                X_train, X_test = X_s[train_index], X_s[test_index]
                y_train, y_test = y_s[train_index], y_s[test_index]
                

            
                test_info = pd.DataFrame(X_test[:,0:2])
                X_train, X_test = X_train[:,5:].astype(np.float64), X_test[:,5:].astype(np.float64)

    
    
                model = XGBClassifier(learning_rate=0.26
                                       ,max_depth=6
                                       ,gamma=1
                                       ,n_estimators=700
                                       ,subsample=0.6
                                       ,colsample_bytree = 0.8
                                       ,n_jobs = -1
                                       ,random_state = 20200926)
                
                model.fit(X_train,y_train)            
                
                y_pred = model.predict(X_test).astype(np.int)
                y_pred_proba = model.predict_proba(X_test)[:,1]
                
                model_name = file_o.replace('_label.csv','')
                
                result_0 = pd.DataFrame()
                result_0['pred_{}'.format(model_name)] = y_pred
                result_0['proba_{}'.format(model_name)] = y_pred_proba
                result_0['fold_{}'.format(model_name)] = fold
                result_0['label_{}'.format(model_name)] = y_test
                
                result_0 = pd.concat([test_info, result_0],axis=1) 
                result_0 = result_0.rename(columns = {0:'index',1:'UniProt'})
                result = pd.concat([result,result_0])
                
                result.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\10CV\10fold_watch.csv",index=False)
                
                os.chdir(r'E:\student\ysq\PCM\Nova\drug-split\train_model')
                pkl_filename = "{}_{}_{}.pkl".format(random_state, model_name, fold)
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(model, file)
            
            
    
            y_pred = result['pred_{}'.format(model_name)]
            y_pred_proba = result['proba_{}'.format(model_name)]
            y_test = result['label_{}'.format(model_name)]
            df = GetMetrics(y_test, y_pred, y_pred_proba)
            df['model'] = model_name
            result_sta = pd.concat([result_sta,df])
            
            result_final = pd.concat([result_final, result], axis=1)
            
    
            result_final.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\10CV\{}_10fold_{}.csv".format(model_name,random_state),index=False)
            result_sta.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\10CV\{}_10fold_sta_{}.csv".format(model_name,random_state),index=False)   
            
        result_final.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\10CV\{}_10fold.csv".format(random_state),index=False)
        result_sta.to_csv(r"E:\student\ysq\PCM\Nova\drug-split\10CV\{}_10fold_sta.csv".format(random_state),index=False)
