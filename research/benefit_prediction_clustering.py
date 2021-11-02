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
              "Spouse Relation", "NoofChildren",
              "PlandesignID", "BenefitTypeID", "BenefitTypeIdentifier", "PlanTypeIdentifier"]

datasets = pd.read_csv("../data/data_plantype2.csv", header=None, names=parameters)
datasets = datasets.drop_duplicates()
columnName = ["AnnualSalary"]

for i in columnName:
    mean_val = datasets[i].mean(skipna=True)
    datasets[i] = datasets[i].fillna(datasets[i].mean())
    datasets[i] = datasets[i].mask(datasets[i] == 0, mean_val)
    datasets[i] = datasets[i].mask(datasets[i] == 1, mean_val)

########################## model for medical
medical_datasets = datasets[datasets["BenefitTypeIdentifier"] == "Medical"]

unUsedIndex = ["PlandesignID", "BenefitTypeID", "BenefitTypeIdentifier"]
medical_datasets = medical_datasets.drop(unUsedIndex, axis=1)

order = ["EmployeeAge", "AnnualSalary", "NoofChildren", "Gender", "CompanyID","Spouse Relation", "MaritalStatus","EmployeeStatus", "PlanTypeIdentifier"]
medical_datasets = medical_datasets[order]

labelencoders = {}

categorical = [ "CompanyID","Gender","Spouse Relation", "MaritalStatus","EmployeeStatus",]

for category in categorical:
    labelencoders[category] = LabelEncoder()
    medical_datasets[category] = labelencoders[category].fit_transform(medical_datasets[category].astype('str').values)

#medical_datasets.sample(frac=1)
X = medical_datasets.iloc[:, :-1].values
Y = medical_datasets.iloc[:, -1].values

onehotencoders = {}

for i in range(len(categorical)):
    onehotencoders[i] = OneHotEncoder(categorical_features=[-1])
    X = onehotencoders[i].fit_transform(X).toarray()
    X = X[:, 1:]



################## clustering
k = len(set(Y))
from sklearn.cluster import KMeans  

kmeans = KMeans(n_clusters=6)  
kmeans.fit(X)  
pred_classes = kmeans.predict(X)

from collections import Counter
n_clusters=k
pred_clusters=kmeans.fit(X).labels_
cluster_labels=[[] for i in range(n_clusters)]
for i, j in enumerate(pred_clusters):
    cluster_labels[j].append(Y[i])
for i in cluster_labels:
    cnt = Counter(i)
    print(cnt)
# print(cluster_labels)
# for cluster in range(k):
#     print('cluster: ', cluster)
#     print(Y[np.where(pred_classes == cluster)])
# import matplotlib.pyplot as plt
# plt.scatter(X[:, -2], X[:, -3], s=50);
# plt.show()
#
# data = {
# "EmployeeStatus":"Active",
# "Gender":"Male",
# "EmployeeAge":52,
# "AnnualSalary":283000,
# "NoofChildren":3,
# "MaritalStatus":"Married",
# "Spouse Relation":"Spouse",
# "CompanyID":723	,
# "EligibilePlanDesignIDs": "25#24-25-33-34-35-92|24#1-2"
# }
#
# predicion_variable = "PlanTypeIdentifier"
# user_features = []
#
# for category in categorical:
#     data[category] = labelencoders[category].transform([str(data[category])])[0]
# order = ["EmployeeAge", "AnnualSalary", "NoofChildren", "Gender", "CompanyID","Spouse Relation", "MaritalStatus","EmployeeStatus", "PlanTypeIdentifier"]
#
# for ord in order:
#     if ord != predicion_variable:
#         user_features.append(data[ord])
# user_features = np.array(user_features).reshape(-1, len(user_features))
#
# for i in range(len(categorical)):
#     user_features = onehotencoders[i].transform(user_features).toarray()
#     user_features = user_features[:, 1:]
#
# # user_features = scaler.transform(user_features)
#
# prob = kmeans.predict(user_features)
# print(prob)
