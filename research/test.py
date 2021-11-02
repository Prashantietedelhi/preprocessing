from recommendation.getdatasets import GetDatasets
from recommendation.preprocessing import Preprocessing
from recommendation.recommendation import Recommendation

getds = GetDatasets()

predicion_variable = "PlanTypeIdentifier"
missingcolumnName = ["AnnualSalary"]
order = ["EmployeeAge", "AnnualSalary", "NoofChildren", "Gender", "EmployeeStatus", "MaritalStatus",
         "Spouse Relation", "CompanyID", "PlanTypeIdentifier"]
order = [i.lower().strip() for i in order]
parameters = ["CompanyID", "Gender", "EmployeeAge",
              "AnnualSalary", "EmployeeStatus", "MaritalStatus",
              "Spouse Relation", "NoofChildren",
              "PlandesignID", "BenefitTypeID", "BenefitTypeIdentifier", "PlanTypeIdentifier"]
parameters = [i.lower().strip() for i in parameters]

unusedIndex = ["CompanyID", "Gender", "EmployeeAge",
               "AnnualSalary", "EmployeeStatus", "MaritalStatus",
               "Spouse Relation", "NoofChildren",
               "PlanTypeIdentifier"]
unusedIndex = [i.lower().strip() for i in unusedIndex]

categorical = ["Gender", "EmployeeStatus", "MaritalStatus", "Spouse Relation", "CompanyID"]
categorical = [i.lower().strip() for i in categorical]

import pandas as pd


# data = pd.read_csv("../../data/data_plantype.csv", header=None, names=parameters)
data = getds.get_Data_CSV("../data/data_plantype.csv",parameters)
data = data.drop_duplicates()

data = data[data["BenefitTypeIdentifier".lower()] == "Medical"]
obj = Preprocessing(data, parameters, unusedIndex, missingcolumnName, categorical, order, predicion_variable)
import numpy as np

data, label = obj.getDataSets()
# label = np.ndarray
print(type(data))
print(type(label))

model = Recommendation(data,label)
print(model.accuracy())


# print(obj.get_output_classes(4))
inputdatasets = {"EmployeeStatus".lower(): "Active",
                 "Gender".lower(): "Male",
                 "EmployeeAge".lower(): 52,
                 "AnnualSalary".lower(): 283000,
                 "NoofChildren".lower(): 3,
                 "MaritalStatus".lower(): "Married",
                 "Spouse Relation".lower(): "Spouse",
                 "PostalCode".lower(): "92630",
                 "CompanyID".lower(): 13641,

                 }
print(model.predict(obj.feature_extraction(inputdatasets)))