# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


parameters = ["CompanyID", "Gender", "EmployeeAge",
              "AnnualSalary", "EmployeeStatus", "MaritalStatus",
              "Spouse Relation", "NoofChildren",
              "PlandesignID", "BenefitTypeID", "BenefitTypeIdentifier", "PlanTypeIdentifier"]

datasets = pd.read_csv("../data/data_plantype2.csv", header=None, names=parameters)
datasets = datasets.drop_duplicates()


print((datasets[['AnnualSalary']]==3).sum())
datasets[['AnnualSalary']] = datasets[['AnnualSalary']].replace(0, np.NaN)
datasets[['AnnualSalary']] = datasets[['AnnualSalary']].replace(1, np.NaN)

print(datasets.isnull().sum())
imputer = Imputer()

datasets.iloc[:,3:4] = imputer.fit_transform(datasets.iloc[:,3:4])

columnName = ["AnnualSalary"]


# datasets = datasets.dropna()
# datasets = datasets[datasets['AnnualSalary']!=0]
# datasets = datasets[datasets['AnnualSalary']!=1]
#for i in columnName:
#    mean_val = datasets[i].mean(skipna=True)
#    datasets[i] = datasets[i].fillna(datasets[i].mean())
#    datasets[i] = datasets[i].mask(datasets[i] == 0, mean_val)
#    datasets[i] = datasets[i].mask(datasets[i] == 1, mean_val)

########################## model for medical
medical_datasets = datasets[datasets["BenefitTypeIdentifier"] == "Medical"]

unUsedIndex = ["PlandesignID", "BenefitTypeID", "BenefitTypeIdentifier"]
medical_datasets = medical_datasets.drop(unUsedIndex, axis=1)

order = ["EmployeeAge", "AnnualSalary", "NoofChildren", "Gender", "EmployeeStatus", "MaritalStatus",
         "Spouse Relation", "CompanyID", "PlanTypeIdentifier"]
medical_datasets = medical_datasets[order]

labelencoders = {}

categorical = ["Gender", "EmployeeStatus", "MaritalStatus", "Spouse Relation", "CompanyID"]

for category in categorical:
    labelencoders[category] = LabelEncoder()
    medical_datasets[category] = labelencoders[category].fit_transform(medical_datasets[category].astype('str').values)

# medical_datasets = medical_datasets.sample(frac=1)
X = medical_datasets.iloc[:, :-1].values
Y = medical_datasets.iloc[:, -1].values

onehotencoders = {}

for i in range(len(categorical)):
    onehotencoders[i] = OneHotEncoder(categorical_features=[-1])
    X = onehotencoders[i].fit_transform(X).toarray()
    X = X[:, 1:]


size = 0.2
random_state = 42
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=size)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


from sklearn.model_selection import cross_val_score
model= RandomForestClassifier(n_estimators =  300, min_samples_split= 10, min_samples_leaf = 2, max_features = 'sqrt', max_depth =  60, bootstrap = True)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=6,  shuffle=True,random_state=42)
result = cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy')
print(result)
print(result.mean())
model.fit(train_x, train_y)
predictions = model.predict(test_x)
acc = accuracy_score(test_y, predictions)
print(acc)

#scaler = StandardScaler()
#X = scaler.fit_transform(X)
######### BEST FEATURE
# X  = scaler.fit_transform(X)
# from sklearn import metrics
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X, Y)
# fec = model.feature_importances_
# for c,f in zip(order,fec):
#     print(c,f)
#


########### TRAINING ALGORITHM


#model = RandomForestClassifier(n_estimators= 200, random_state=random_state)
#
#
#
# model.fit(train_x, train_y)
# #
# predictions = model.predict(test_x)
# acc = accuracy_score(test_y, predictions)
# print(acc)

#
# '''
# from sklearn.model_selection import RandomizedSearchCV
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf =RandomForestClassifier()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(train_x, train_y)
# print(rf_random.best_params_)
# '''
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(model,X,Y, cv=kfold)
#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# ##################### PREDICTION NEW DATA
# # data = {
# # "EmployeeStatus":"Active",
# # "Gender":"Male",
# # "EmployeeAge":52,
# # "AnnualSalary":283000,
# # "NoofChildren":3,
# # "MaritalStatus":"Married",
# # "Spouse Relation":"Spouse",
# # "CompanyID":723	,
# # "EligibilePlanDesignIDs": "25#24-25-33-34-35-92|24#1-2"
# # }
# #
# # predicion_variable = "PlanTypeIdentifier"
# # user_features = []
# #
# # for category in categorical:
# #     data[category] = labelencoders[category].transform([str(data[category])])[0]
# # order = ["EmployeeAge","AnnualSalary","NoofChildren","Gender","EmployeeStatus","MaritalStatus","Spouse Relation","CompanyID","PlanTypeIdentifier"]
# # for ord in order:
# #     if ord != predicion_variable:
# #         user_features.append(data[ord])
# # user_features = np.array(user_features).reshape(-1, len(user_features))
# #
# # for i in range(len(categorical)):
# #     user_features = onehotencoders[i].transform(user_features).toarray()
# #     user_features = user_features[:, 1:]
# #
# # user_features = scaler.transform(user_features)
# #
# # classes = model.classes_
# # prob = model.predict_proba(user_features)
# # res = dict(zip(classes,prob[0]))
# # print(res)
#
# ######################## BEST PARAMETER SELECTION
# '''classifier = RandomForestClassifier(random_state=42)
# param_dist = {'max_depth': [2, 3, 4, 5, None],
#               'bootstrap': [True, False],
#               'max_features': ['auto', 'sqrt', 'log2', None],
#               'criterion': ['gini', 'entropy'],
#               'n_estimators': [10,50,100,500]}
# cv_rf = GridSearchCV(classifier, cv=10,
#                      param_grid=param_dist,
#                      )
# cv_rf.fit(X, Y)
# print('Best Parameters using grid search: \n', cv_rf.best_params_)
# '''