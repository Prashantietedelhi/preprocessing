# Preprocessing module
# Authors: <Singh, Prashant <Prashant.Singh@careerbuilder.com>>
'''
This module is responsible for all the pre processing needed on the data before feeding it to machine learning algorithms.

It does following things:
1. Label encode/transform the non-numerical labels (as long as they are hashable and comparable) to numerical labels.
2. Imputer - Replace missing values with mean of the column.
3. One Hot Encoding: Encoding the label encoded object into one hot encode.
e.x:

'''
import sys,os
curpath = os.path.join(os.path.dirname(os.path.realpath("__file__")), "..")
sys.path.insert(0,curpath)

from common.get_logger import GetLogger

from common.getdatasets import GetDatasets

import configparser
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

####################### Config file reading
config_file_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","..","config","config2.cfg")

config_obj = configparser.ConfigParser()

try:
    config_obj.read(config_file_loc)
    debugLevel = int(config_obj.get("PreProcessing", "debuglevel"))
    logfilename = config_obj.get("PreProcessing", "logfilename")
    filename = config_obj.get("Common", "filename")
    index_conf = config_obj.get("Common", "index")
    categorical = config_obj.get("Common", "categorical")
    usedIndex = config_obj.get("Common", "used_index")
    missingColumns = config_obj.get("Common", "missingColumns")
    features = config_obj.get("Common", "features")
    order = config_obj.get("Common", "order")
    query = config_obj.get("Common", "query")

except Exception as e:
    raise Exception("Config file reading error: "+str(e))

####################### Logging Functionality
logfilename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","..","logs",logfilename)
loggerobj = GetLogger("PlanPreprocessing",logfilename,debugLevel)
logger = loggerobj.getlogger()
####################### Initialization

try:
    index = index_conf.split(",")
    index = [i.lower().strip() for i in index]
except Exception as e:
    logger.error("Failed to extract index :"+str(index)+" REASON: "+str(e))
    raise Exception("Failed to extract index :"+str(index))

try:
    categorical = categorical.split(",")
    categorical = [i.lower().strip() for i in categorical if i.lower().strip()!='']
except Exception as e:
    logger.error("Failed to extract categorical variables :"+str(categorical)+" REASON : "+str(e))
    raise Exception("Failed to extract categorical variables :"+str(categorical))

try:
    usedIndex = usedIndex.split(",")
    usedIndex = [i.lower().strip() for i in usedIndex]
except Exception as e:
    logger.error("Failed to extract used_index variables :"+str(usedIndex)+" REASON : "+str(e))
    raise Exception("Failed to extract used_index variables :"+str(usedIndex))

try:
    missingColumns = missingColumns.split(",")
    missingColumns = [i.lower().strip() for i in missingColumns]
except Exception as e:
    logger.error("Failed to extract missingColumns variables :"+str(missingColumns)+" REASON : "+str(e))
    raise Exception("Failed to extract missingColumns variables :"+str(missingColumns))

try:
    features = features.split(",")
    features = [i.strip().lower() for i in features]
except Exception as e:
    logger.error("Failed to extract the features from config file : "+" REASON : "+str(e))
    raise Exception("Failed to extract the features from config file")

try:
    order = order.split(",")
    order = [i.strip().lower() for i in order]
except Exception as e:
    logger.error("Failed to extract the order from config file : " + " REASON : " + str(e))
    raise Exception("Failed to extract the order from config file")

if len(set(missingColumns) - set(index)) != 0 or len(set(usedIndex) - set(index)) != 0 :
    logger.error("Incorrect data either in index or categorical or used_index or missingColumns. Check config file")
    raise Exception("Incorrect data either in index or categorical or used_index or missingColumns. Check config file")



logger.info("PlanPreprocessing Library")

class PlanPreprocessing():
    '''
    This class is responsible to do all the pre processing step needed.
    The following pre processing steps are done:
    1. Label Encoding or transformation of non numerical labels to numerical labels
    2. Replace missing or 0's values with mean of the column
    3. One Hot encoding of Label encoded values
    The following functions are defined:
    __init__: does all initialization.
    __getDatasets__: get the datasets from the source.
    __labelEncode__: Encode the categorical variables into numeric variables.
    __imputer__ : impute missing values with mean of the column.
    __replaceZeroSalary__: replace zeros salary with mean of the salary.
    __oneHotEncoding_fit_transform__: fits and transform the training data.
    __oneHotEncoding_transform__: one hot encode the new feature
    feature_extraction: extract the features from the new data point
    '''
    def __init__(self):
        '''
            Constructor: It will initialize variables
        '''

        self.getDataSetsObj = GetDatasets()
        self.datasets = None
        self.encoding_objs = {}
        self.oneHotEncodingObjs = {}
    def __getDatasets__(self):
        '''
        This function does the following:
            1. gets the data set from the getdatasets library
            2. remove the unused columns from the datasets
            3. replace the columns having Nan or 0 value with mean of the column.
            4. label encode the categorical variables into numeric value.
            5. one hot encode the label encoded columns
        :return:
        '''
        #################### gets the data set from the getdatasets library
        try:
            self.datasets = self.getDataSetsObj.get_Data_SQL(index,query)
            self.getDataSetsObj.save_csv(self.datasets)
            self.datasets =self.datasets.dropna()
        except Exception as e:
            logger.error(e)
            raise Exception(e)

        ################### remove the unused columns from the datasets
        try:
            unUsedIndex = list(set(index) - set(usedIndex))
            self.datasets = self.datasets.drop(unUsedIndex, axis=1)
            self.datasets = self.datasets.drop_duplicates()
        except Exception as e:
            logger.error(e)
            raise Exception(e)

        self.mean_val_columns = {}
        ################### replace the columns having Nan or 0 value with mean of the column.
        try:
            for missingColumn in missingColumns:
                self.__imputer__(missingColumn)
                self.__replaceZeroSalary__(missingColumn)
                # self.__replace_val__(missingColumn)
        except Exception as e:
            logger.error(e)
            raise Exception(e)

        ################## label encode the categorical variables into numeric value.
        try:
            self.__labelEncode__()
        except Exception as e:
            logger.error(e)
            raise Exception(e)

        self.datasets = self.datasets[order]

        ################## one hot encode the label encoded columns
        try:
            self.__oneHotEncoding_fit_transform__()
        except Exception as e:
            logger.error(e)
            raise Exception(e)
    def __replace_val__(self, columnName):
        self.datasets = self.datasets.dropna()
        self.datasets = self.datasets[self.datasets[columnName] != 0]
        self.datasets = self.datasets[self.datasets[columnName] != 1]
    def __labelEncode__(self):
        '''
        This function is responsible for label encoding or transformation of non numerical labels to numerical labels
        :return:
        '''
        if categorical!=None and len(categorical)>0:
            for c in categorical:
                if c in list(self.datasets.columns):
                    self.encoding_objs[c] = LabelEncoder()
                    self.datasets[c] = self.encoding_objs[c].fit_transform(self.datasets[c].astype('str').values)

    def __imputer__(self, columnName ):
        '''
         Replace missing or 0's values with mean of the column
        :return:
        '''
        ind = list(self.datasets.columns)
        columnIndex = ind.index(columnName)
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

        try:
            self.imputer = self.imputer.fit(self.datasets.iloc[:, columnIndex:columnIndex+1])
        except Exception as e:
            logger.error(e)
            raise Exception(e)

        try:
            self.datasets.iloc[:, columnIndex:columnIndex+1] = self.imputer.transform(self.datasets.iloc[:, columnIndex:columnIndex+1])
        except Exception as e:
            logger.error(e)
            raise Exception(e)

        # self.datasets = self.datasets[~np.isnan(self.datasets).any(axis=1)]

    def __replaceZeroSalary__(self, columnName):
        '''
        Replace zero value with mean of the column
        :param columnName: column name as str type.
        :return:
        '''
        try:
            mean_val = self.datasets[columnName.lower()].mean(skipna=True)
            self.mean_val_columns[columnName.lower()] = mean_val
        except Exception as e:
            logger.error(e)
            raise Exception(e)

        try:
            self.datasets[columnName.lower()] = self.datasets[columnName.lower()].mask(self.datasets[columnName.lower()] == 0,
                                                                               mean_val)
        except Exception as e:
            logger.error(e)
            raise Exception(e)


    def __oneHotEncoding_fit_transform__(self):
        '''
        fits and transform  one hot encoded value of training data
        :return:
        '''
        for i in range(len(categorical)):
            self.oneHotEncodingObjs[i] = OneHotEncoder(categorical_features=[-1])
            self.datasets = self.oneHotEncodingObjs[i].fit_transform(self.datasets).toarray()
            self.datasets = self.datasets[:, 1:]

    def __oneHotEncoding_transform__(self, X):
        '''
        transform into one hot enoded value of training data
        :param X: user label encoded features of numpy array type
        :return: one hot encoded features
        '''
        for i in range(len(categorical)):
            X = self.oneHotEncodingObjs[i].transform(X).toarray()
            X = X[:,1:]
        return X

    def getDataSets(self):
        '''
        calls the getdatasets library to get the datasets and convert into pandas Dataframe
        :return:
        '''
        self.__getDatasets__()
        return pd.DataFrame(self.datasets)

    def feature_extraction(self, user_feature_set):
        '''
        extract the features of the new unseen data.
        :param user_feature_set: dict type having the user demographic information.
        :return: feature point of type numpy array
        '''
        for missingColumn in missingColumns:
            if user_feature_set[missingColumn.lower()] ==0 or user_feature_set[missingColumn.lower()] == None or user_feature_set[missingColumn.lower()] == 'NaN':
                user_feature_set[missingColumn.lower()] = self.mean_val_columns[missingColumn.lower()]
            try:
                int(user_feature_set[missingColumn.lower()])
            except:
                user_feature_set[missingColumn.lower()] = self.mean_val_columns[missingColumn.lower()]
        try:
            for c in categorical:
                try:
                    user_feature_set[c] = self.encoding_objs[c].transform([str(user_feature_set[c])])[0]
                except Exception as e:
                    try:
                        float(user_feature_set[c])
                        user_feature_set[c] = self.encoding_objs[c].transform([str(user_feature_set[c]) + ".0"])[0]
                    except:
                        raise Exception(e)

            features_X = []
            for k in order:
                features_X.append(user_feature_set[k])

            features_X = np.array(features_X).reshape(-1, len(features_X))

            features_X = self.__oneHotEncoding_transform__(features_X)

        except Exception as e:
            logger.error(e)
            raise Exception(e)
        return features_X

if __name__ == "__main__":
    obj = PlanPreprocessing()
#     import numpy as np
    data = obj.getDataSets()
#     # print(data)
#     inputdatasets = {"EmployeeStatus".lower(): "Active",
#                      "Gender".lower(): "Male",
#                      "EmployeeAge".lower(): 52,
#                      "AnnualSalary".lower(): 283000,
#                      "NoofChildren".lower(): 3,
#                      "MaritalStatus".lower(): "Married",
#                      "Spouse Relation".lower(): "Spouse",
#                      "PostalCode".lower(): "92630",
#                      "CompanyID".lower(): 12037,
#                      "BenefittypeID".lower(): 1,
#                      "plandesignid".lower(): 30
#                      }
#     print(obj.feature_extraction(inputdatasets)[0][-4])








