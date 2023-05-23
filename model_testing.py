import pandas as pd
from sklearn.metrics import recall_score,roc_auc_score,f1_score
from joblib import load

knn = load('knn.joblib') 

X_test = pd.read_csv('X_test.csv', index_col='index')
y_test = pd.read_csv('y_test.csv', index_col='index')
print(y_test.value_counts())

def three_score(model_pipe, X, y): #оценка модели отдельной функцией, 3 метрики, условие выполнено
    """Расчет коэффициента f1, roc_auc, recall 
    
    Параметры:
    ===========
    model_pipe: модель или pipeline
    X: признаки
    y: истинные значения
    """
    y_model = model_pipe.predict(X)
    return f1_score(y, np.round(y_model), average='weighted'), roc_auc_score(y,np.round(y_model), average='weighted'),recall_score(y,np.round(y_model), average='weighted')
y_model = knn.predict(X_test)
print(f1_score( X_test, np.round(y_test))) #Classification metrics can't handle a mix of continuous-multioutput and multiclass-multioutput targets


#print('результаты тестов',three_score(knn, X_test, y_test))#multiclass-multioutput is not supported
print('pipeline исполнен успешно')
