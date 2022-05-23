from sklearn.model_selection import KFold,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,auc,roc_auc_score
from sklearn.metrics import precision_score,recall_score
import pandas as pd
import os
from xgboost import XGBClassifier
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold



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
   

    relation = pd.read_csv(r"E:\student\ysq\PCM\data\next_data\relation_label.csv")
#    test = pd.read_csv(r"E:\student\ysq\PCM\data\next_data\test.csv")
#    relation = Constract(relation, test)
#    group = relation.apply(lambda x : x['UniProt'] + '-' + str(x['label']),axis=1)
#    group = np.array(group)
    
    
    os.chdir(r"E:\student\ysq\PCM\data\mol_prot\Withlabel")
    files = os.listdir()
    
    random_states = [20200916]
#    [20200917,20200918,20200919,20200920,20200921, 20200922]
    for random_state in random_states:
        
        result_final = pd.DataFrame()
        result_sta = pd.DataFrame()
        
        for file_o in files:
            os.chdir(r"E:\student\ysq\PCM\data\mol_prot\Withlabel")
            data = pd.read_csv(file_o)
            data = data.rename(columns = {'0':'index','1':'UniProt'})
#            data = Constract(data, test)
    
            X = np.array(data.iloc[:,:-1])
            y = np.array(data.iloc[:,-1])
            
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
            
            
            result = pd.DataFrame()
            fold = 0
            for train_index, test_index in skf.split(X, y):
                fold += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
    
                test_info = pd.DataFrame(X_test[:,0:2])
                X_train, X_test = X_train[:,5:].astype(np.float64), X_test[:,5:].astype(np.float64)
            
                model = XGBClassifier(learning_rate=0.3
                                       ,max_depth=6
                                       ,n_estimators=500
                                       ,n_jobs = 16
                                       ,random_state=20200910)
            
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
                
                result.to_csv(r"E:\student\ysq\PCM\Nova\proba\test\10fold_watch.csv",index=False)
                
                os.chdir(r'E:\student\ysq\PCM\Nova\train_model\test_model')
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
            
    
    
            result_final.to_csv(r"E:\student\ysq\PCM\Nova\proba\test\{}_10fold_{}.csv".format(model_name,random_state),index=False)
            result_sta.to_csv(r"E:\student\ysq\PCM\Nova\proba\test\{}_10fold_sta_{}.csv".format(model_name,random_state),index=False)   
            
        result_final.to_csv(r"E:\student\ysq\PCM\Nova\proba\test\{}_10fold.csv".format(random_state),index=False)
        result_sta.to_csv(r"E:\student\ysq\PCM\Nova\proba\test\{}_10fold_sta.csv".format(random_state),index=False)
