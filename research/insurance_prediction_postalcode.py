# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
parameters = ["CompanyID", "Gender", "EmployeeAge",
              "AnnualSalary", "EmployeeStatus", "MaritalStatus",
              "Spouse Relation", "NoofChildren","PostalCode",
              "PlandesignID", "BenefitTypeID", "BenefitTypeIdentifier", "PlanTypeIdentifier"]

datasets = pd.read_csv("../data/data_plantype_postalcode.csv", header=None, names=parameters)
datasets = datasets.drop_duplicates()
columnName = ["AnnualSalary"]

for i in columnName:
    mean_val = datasets[i].mean(skipna=True)
    datasets[i] = datasets[i].fillna(datasets[i].mean())
    datasets[i] = datasets[i].mask(datasets[i] == 0, mean_val)
    datasets[i] = datasets[i].mask(datasets[i] == 1, mean_val)

########################## model for medical
medical_datasets = datasets[datasets["BenefitTypeIdentifier"] == "Medical"]

unUsedIndex = ["CompanyID","PlandesignID", "BenefitTypeID", "BenefitTypeIdentifier"]
medical_datasets = medical_datasets.drop(unUsedIndex, axis=1)

order = ["EmployeeAge", "AnnualSalary", "NoofChildren", "Gender", "EmployeeStatus", "MaritalStatus",
         "Spouse Relation","PostalCode", "PlanTypeIdentifier"]
medical_datasets = medical_datasets[order]

labelencoders = {}

categorical = ["Gender", "EmployeeStatus", "MaritalStatus", "Spouse Relation",  "PostalCode"]

for category in categorical:
    labelencoders[category] = LabelEncoder()
    medical_datasets[category] = labelencoders[category].fit_transform(medical_datasets[category].astype('str').values)

X = medical_datasets.iloc[:, :-1].values
Y = medical_datasets.iloc[:, -1].values

onehotencoders = {}

for i in range(len(categorical)):
    onehotencoders[i] = OneHotEncoder(categorical_features=[-1])
    X = onehotencoders[i].fit_transform(X).toarray()
    X = X[:, 1:]

size = 0.2
random_state = 42
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=size, random_state=random_state)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

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
model = RandomForestClassifier(n_estimators= 10, random_state=42)
model.fit(train_x, train_y)

predictions = model.predict(test_x)
acc = accuracy_score(test_y, predictions)
print(acc)


##################### PREDICTION NEW DATA
data = {
"EmployeeStatus":"Active",
"Gender":"Male",
"EmployeeAge":37,
"AnnualSalary":78239,
"NoofChildren":2,
"MaritalStatus":"Married",
"Spouse Relation":"Spouse",
"CompanyID":13641	,
"PostalCode" :"91762",
"EligibilePlanDesignIDs": "25#24-25-33-34-35-92|24#1-2"
}

predicion_variable = "PlanTypeIdentifier"
user_features = []

for category in categorical:
    data[category] = labelencoders[category].transform([str(data[category])])[0]
order = ["EmployeeAge","AnnualSalary","NoofChildren","Gender","EmployeeStatus","MaritalStatus","Spouse Relation", "PostalCode","PlanTypeIdentifier"]
for ord in order:
    if ord != predicion_variable:
        user_features.append(data[ord])
user_features = np.array(user_features).reshape(-1, len(user_features))

for i in range(len(categorical)):
    user_features = onehotencoders[i].transform(user_features).toarray()
    user_features = user_features[:, 1:]

user_features = scaler.transform(user_features)

classes = model.classes_
prob = model.predict_proba(user_features)
res = dict(zip(classes,prob[0]))
print(res)
######################## BEST PARAMETER SELECTION
# classifier = RandomForestClassifier(random_state=42)
# param_dist = {'max_depth': [2, 3, 4, None],
#               'bootstrap': [True, False],
#               'max_features': ['auto', 'sqrt', 'log2', None],
#               'criterion': ['gini', 'entropy'],
#               'n_estimators': [10, 50, 100, 500]}
# cv_rf = GridSearchCV(classifier, cv=10,
#                      param_grid=param_dist,
#                      )
# cv_rf.fit(train_x, train_y)
# print('Best Parameters using grid search: \n', cv_rf.best_params_)