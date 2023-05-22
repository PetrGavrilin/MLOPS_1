import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline#импортирую бибилотеку трубы
from sklearn.impute import SimpleImputer# Одномерный импутер для заполнения пропущенных значений с помощью простых стратегий. 
from sklearn.model_selection import train_test_split
df = pd.read_csv('df.csv')
df = df.drop(['state'],axis=1)

df['international_plan'] = df['international_plan'].map({'yes': 1, 'no': 0})#провожу перекодировку признаков
df['voice_mail_plan'] = df['voice_mail_plan'].map({'yes': 1, 'no': 0})
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})
df['area_code'] = df['area_code'].map({'area_code_415': 1, 'area_code_408': 0,'area_code_510':3})

X_train, X_test, y_train, y_test = train_test_split(df.drop(['churn'],axis=1), df['churn'].ravel(), stratify=df['churn'], test_size=0.2,random_state=0) 

standart_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    
    ('scaler', StandardScaler())
])

X_train = standart_pipe.fit_transform(X_train)# проводим обучение
X_test = standart_pipe.fit_transform(X_test)# проводим обучение

X_train = pd.DataFrame(X_train)#иначе в csv не запихать
X_test = pd.DataFrame(X_test)

X_test.to_csv('X_test.csv', index_label='index')
y_test.to_csv('y_test.csv', index_label='index') 
X_train.to_csv('X_train.csv', index_label='index')
y_train.to_csv('y_train.csv', index_label='index') 
print('data_preprocessing исполнен')
