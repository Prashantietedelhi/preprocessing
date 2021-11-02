# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
#import csv
#csvw = csv.writer(open("Accuracy_Plan.csv","w",newline=''))
#csvw.writerow(("ComapnyID","BenefitID","Num Datapoints","Num Labels","Accuracy"))
st="CompanyID,Gender,EmployeeAge,AnnualSalary,EmployeeStatus,MaritalStatus,Spouse Relation,NoofChildren,PlandesignID,BenefitTypeID,BenefitTypeIdentifier,PlanTypeIdentifier"
used_index="CompanyID,Gender,EmployeeAge,AnnualSalary,EmployeeStatus,MaritalStatus,Spouse Relation,NoofChildren,PlandesignID,BenefitTypeID"
categorical="Gender,MaritalStatus,Spouse Relation,EmployeeStatus"
order = "EmployeeAge,AnnualSalary,NoofChildren,BenefitTypeID,PlandesignID,CompanyID,EmployeeStatus,MaritalStatus,Spouse Relation,Gender"

used_index = used_index.split(",")
parameters =st.split(",")
fileloc = "../data/data_plantype2.csv"
datasets = pd.read_csv(fileloc, header = None, names = parameters)

unUsedIndex = list(set(parameters) - set(used_index))

datasets = datasets.drop(unUsedIndex, axis=1)

mean_val_columns={}
missingColumns="AnnualSalary"
missingColumns = missingColumns.split(",")
for columnName in missingColumns:
    mean_val = datasets[columnName].mean(skipna=True)
    mean_val_columns[columnName] = mean_val
    datasets[columnName] = datasets[columnName].mask(datasets[columnName] == 0,
                                                                               mean_val)
    
    ind = list(datasets.columns)
    columnIndex = ind.index(columnName)
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(datasets.iloc[:, columnIndex:columnIndex+1])
    datasets.iloc[:, columnIndex:columnIndex+1] = imputer.transform(datasets.iloc[:, columnIndex:columnIndex+1])
    
    
encoding_objs = {}

categorical = categorical.split(",")

if categorical!=None and len(categorical)>0:
    for c in categorical:
        if c in list(datasets.columns):
            encoding_objs[c] = LabelEncoder()
            datasets[c] = encoding_objs[c].fit_transform(datasets[c].astype('str').values)
            
            
order= order.split(",")

datasets = datasets[order]

oneHotEncodingObjs={}

for i in range(len(categorical)):
    oneHotEncodingObjs[i] = OneHotEncoder(categorical_features=[-1])
    datasets = oneHotEncodingObjs[i].fit_transform(datasets).toarray()
    datasets = datasets[:, 1:]
    
    
allDataSets = pd.DataFrame(datasets)


companyID = 14055
BenefitTypeID = 2
compyids = list(set(list(allDataSets.iloc[:,-1].values.astype("int"))))
allacc = []
for companyID in compyids:
    
    datasets_1 = allDataSets[allDataSets.iloc[:,-1] == companyID]
    allval = datasets_1.iloc[:,-3].astype(int)
    allval = list(set(list(allval)))
    for BenefitTypeID in allval:
        datasets_2 = datasets_1[datasets_1.iloc[:,-3] == BenefitTypeID]
        
        
        Y = datasets_2.iloc[:,-2].values
        X = datasets_2.iloc[:, :-3].values
        #if len(list(set(Y)))>1:
        random_state=42
        size = 0.2
        train_x, test_x, train_y, test_y = train_test_split(X, Y,test_size=size, random_state=random_state)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        
        model = RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=4, max_features='auto',
                                               n_estimators=50, random_state=42)
        
        model.fit(train_x, train_y.astype(int))
        predictions = model.predict(test_x)
        acc = accuracy_score(test_y.astype(int), predictions)
        print(acc)
        allacc.append(acc)
        #csvw.writerow((companyID, BenefitTypeID,len(list(Y)),len(list(set(Y))), acc))
print(sum(allacc)/len(allacc))