#!/usr/bin/env python
# coding: utf-8

# In[1]:


#kutuphaneleri importladim
from sklearn import datasets
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
#datayi yukledim
wine = datasets.load_wine()

#iki sinifa ayirdim
X = wine.data
Y = wine.target

#test ve traine boldum
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


names = []
models = []
score = 'accuracy'
results = []


models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=2))),
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=score)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())


    print(msg)


# In[ ]:




