from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import csv

from sklearn.preprocessing import  OneHotEncoder

index = "CompanyID,Gender,EmployeeAge,AnnualSalary,EmployeeStatus,MaritalStatus,Spouse Relation,NoofChildren,PlandesignID,BenefitTypeID,BenefitTypeIdentifier,PlanTypeIdentifier,CoverageAmount,CoverageAmountDeterminationTypeID,CoverageAmountDeterminationTypeName,StartAmount,EndAmount,IncrementFactor,MaximumCoverageFlatAmount,Benefittemplateid"
usedIndex ="Gender,EmployeeAge,AnnualSalary,EmployeeStatus,MaritalStatus,Spouse Relation,NoofChildren,PlanTypeIdentifier"
categorical = "Gender,MaritalStatus,Spouse Relation,EmployeeStatus"
order = "EmployeeAge,AnnualSalary,NoofChildren,Gender,EmployeeStatus,MaritalStatus,Spouse Relation,PlanTypeIdentifier"
missingColumns="AnnualSalary"

index = index.split(",")
index = [i.strip().lower() for i in index]

missingColumns = missingColumns.split(",")
missingColumns = [i.strip().lower() for i in missingColumns]


categorical = categorical.split(",")
categorical = [i.strip().lower() for i in categorical]


order = order.split(",")
order = [i.strip().lower() for i in order]

usedIndex = usedIndex.split(",")
usedIndex = [i.strip().lower() for i in usedIndex]


unUsedIndex = list(set(index) - set(usedIndex))



datasets_temp = pd.read_csv("data_plantype2.csv",header=None, names = index)
datasets_temp = datasets_temp.drop_duplicates()



plans = list(set(datasets_temp['BenefitTypeIdentifier'.lower()]))
plans  = ["Dental",]
#csvwriterobj = csv.writer(open("benefit_accuracy_without_companyid2.csv","w",newline=''))
plan = plans[0]

#for plan in plans:
    #plans = "Medical"
    
datasets = datasets_temp[datasets_temp["BenefitTypeIdentifier".lower()] == plan]
datasets = datasets.drop(unUsedIndex, axis=1)

for columnName in missingColumns:
    ind = list(datasets.columns)
    columnIndex = ind.index(columnName)
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

    imputer = imputer.fit(datasets.iloc[:, columnIndex:columnIndex+1])
    

   
    datasets.iloc[:, columnIndex:columnIndex+1] =imputer.transform(datasets.iloc[:, columnIndex:columnIndex+1])
    
    
    mean_val = datasets[columnName.lower()].mean(skipna=True)
   
    datasets[columnName] = datasets[columnName.lower()].mask(datasets[columnName.lower()] == 0,
                                                                               mean_val)

    datasets[columnName] = datasets[columnName.lower()].mask(datasets[columnName.lower()] == 1,
                                                                               mean_val)
    
encoding_objs = {}
if categorical!=None and len(categorical)>0:
    for c in categorical:
        if c in list(datasets.columns):
            encoding_objs[c] = LabelEncoder()
            datasets[c] = encoding_objs[c].fit_transform(datasets[c].astype('str').values)


datasets = datasets[order]

X = datasets.iloc[:,:-1].values
Y = datasets.iloc[:,-1].values    

oneHotEncodingObjs = {}
for i in range(len(categorical)):
    oneHotEncodingObjs[i] = OneHotEncoder(categorical_features=[-1])
    X = oneHotEncodingObjs[i].fit_transform(X).toarray()
    X = X[:, 1:]
    
    
model = RandomForestClassifier(random_state=42, n_estimators =  300, min_samples_split= 10, min_samples_leaf = 2, max_features = 'sqrt', max_depth =  60, bootstrap = True)
model.fit(X, Y)
#for feature in zip(order, model.feature_importances_):
#    print(feature)

# cv_rf = GridSearchCV(classifier, cv=10,
#                       param_grid=param_dist,
#                       )
# cv_rf.fit(X, Y)
# print('Best Parameters using grid search: \n', cv_rf.best_params_)
    
kfold = KFold(n_splits=6, shuffle=True, random_state=42)
result = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
print(result)
print(plan, result.mean())
#csvwriterobj.writerow((plan, result.mean(), len(Y)))