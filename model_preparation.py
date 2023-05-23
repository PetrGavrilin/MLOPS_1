import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.pipeline import Pipeline#импортирую бибилотеку трубы


X_train = pd.read_csv('X_train.csv', index_col='index')
y_train_pd = pd.read_csv('y_train.csv', index_col='index')

y_train = y_train_pd[0]


pipe = Pipeline([ 
    ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=2)) # назначаем в качестве модели knn
])

pipe = pipe.fit(X_train, y_train)

dump(pipe, 'knn.joblib') #задаём название для модели

pipe_knn = load('knn.joblib') #назначаем переменную и сохраняем
print('model_preparation исполнен')
