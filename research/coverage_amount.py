import csv
import pandas as pd

coveragePlanData = pd.read_csv("../data/coverage_plans.csv")
col = ["DBServer","DBName","CompanyID","CompanyNAme","EmployeeId","Gender","EmployeeAge","GrossAnnualSalary","EmployeeStatus","MArtialStatus","SpouseRelationship","ChildCount","PostalCode","PlandesignID","PlanDescription","BenifitTypeId","BenifitName","PlanType","BenifiProviderID","BenifitProviderName","HealthTierAbbriviation","ApprovedCoverageAmount","EmployeePreTax","EmployeePostTax","EmployeerPreTax","EmployerPostTax","BenifirTypeIdentifierNAme","PlanTypeIdentifier"]

data = pd.read_csv("../data/query_results_3lakhs.csv",header=None,names=col)

usedindex = "EmployeeStatus,Gender,EmployeeAge,GrossAnnualSalary,ChildCount,MartialStatus,SpouseRelationship,CompanyID,ApprovedCoverageAmount,PlandesignID"

usedindex=usedindex.split(",")

unusedindex = list(set(col)-set(usedindex))
data = data.drop(unusedindex, axis=1)


company14074 = coveragePlanData[coveragePlanData.CompanyID==14074]
data14074 = data[data.CompanyID==14074]
data14074 = data14074[data14074.PlandesignID.isin(list(company14074.PlanDesignID))]
data14074_97  = data14074[data14074.PlandesignID == 97]


data14074_97 = data14074_97.drop(["CompanyID","PlandesignID"], axis =1)


X = data14074_97.iloc[:,[0,1,2,5]].values
Y = data14074_97.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

categorical = [0]
non_categorical = [1,2,3]
preprocess = make_column_transformer(
        
    (non_categorical, StandardScaler())
    
)

X = preprocess.fit_transform(X)

from sklearn.model_selection import train_test_split, GridSearchCV
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state = 42,n_estimators =  50, min_samples_split= 10, min_samples_leaf = 2, max_features = 'sqrt', max_depth =  60, bootstrap = True)

from sklearn.svm import SVC
model= SVC()
model.fit(train_x, train_y)
#
predictions = model.predict(test_x)
from sklearn.metrics import accuracy_score

acc = accuracy_score(test_y, predictions)

'''
coveragePlanData = coveragePlanData.set_index(coveragePlanData.CoverageAmountDeterminationType)

coverageplan1  = coveragePlanData.loc[2]
coverageplan2 =  coveragePlanData.loc[4]

coverageplan1Data = data[data[0].isin(coverageplan1.CompanyID)]
coverageplan2Data = data[data[0].isin(coverageplan2.CompanyID)]


list(set(coveragePlanData.CompanyID))


dropcolumns = [0,1,3,4,12,14,16]
'''
ds = company14074[["StartAmount","EndAmount","IncrementalFactor"]]

ds_st = ds["EndAmount"].values

avg = sum(ds_st)/len(ds_st)



coverage_2 = coveragePlanData[coveragePlanData.CoverageAmountDeterminationType == 2]
coverage_2["newfield"] = coverage_2.