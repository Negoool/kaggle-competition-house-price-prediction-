'''kaggle Competition: house price prediction Aug18'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

import os
os.system('clear')


def cat_num_list(X):
    ''' seperate numerical and categorical features based on their type'''
    cat = []
    num = []
    for col in  list(X.columns):
        if (X[col].dtype) != ('object'):
            num.append(col)
        else :
            cat.append(col)
    # MSSubClass is categorical with 20,40 as values
    num.remove('MSSubClass')
    cat.append('MSSubClass')
    return num, cat

class none_to_category(BaseEstimator, TransformerMixin):
    ''' for some categorival attributes None  means it does not have the feature
    this class fill None with DONT '''
    def __init__(self, attr_list = None):
        self.attr_list = attr_list
    def fit(self, X, y = None):
        return self
    def transform(self, X, Y = None):
        if self.attr_list is not None:
            X[self.attr_list] = X[self.attr_list].fillna(0)
            return X
        else:
            return X

class my_Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        l = {}
        for col in list(X.columns):
            l[col] = (X[col].value_counts().index[0])
        self.fillingvalue = pd.Series(l)
        return self

    def transform(self, X, y = None):
        return X.fillna(self.fillingvalue)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    ''' choose subset of data(seperate categorical and numerical features) and\
    convert to numpy array'''
    def __init__(self, attr_list):
        self.attr_list = attr_list

    def fit(self, X,y = None):
        if self.attr_list is None:
            raise ValueError('attribue list is empty')
        return self

    def transform(self, X, y = None):
        return X[self.attr_list]

class to_numpy(BaseEstimator, TransformerMixin):
    def fit(self, X , y = None):
        return self
    def transform(self, X, y = None):
        return X.values


# read CSV file in Dataframe
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
#data_train.info()

data = data_train.copy()
# seperate X nad y
y_train = data['SalePrice']
X_train = data.drop(['Id','SalePrice'], axis = 1)

### handling missing data
# number of missing values for each feature
print X_train.isnull().sum(axis = 0).sort_values(ascending = False)
# there are three groups
# 1)not having a facility :custom_written class(none_to_category)
# 2)real missing in numerical data: sklearn imputer
# 3)real misssing in categorical data: custom_written imputer

#1) find  attributes(categorivcal) that missing means it does not have  facility
attr_missing = list(data.isnull().sum(axis = 0).sort_values(ascending = False)\
.index.values)[:19]
att_missing_1 = [attr for attr in attr_missing if attr  not in \
('LotFrontage', 'Electrical')]

# custom witten function for fing categorical and numerical features
num_features, cat_features =  cat_num_list(X_train)

my_obj = none_to_category(attr_list = att_missing_1)
X_train = my_obj.fit_transform(X_train)


num_pipeline = Pipeline([\
('num_selector',DataFrameSelector(num_features)),\
('to_numpy',to_numpy()),\
('imputer', Imputer(strategy = 'median', verbose = 2)),\
])

X_train_num = num_pipeline.fit_transform(X_train)
print X_train_num.shape
#
cat_pipeline = Pipeline([\
('cat_selector', DataFrameSelector(cat_features)),
('imputer', my_Imputer()),
('to_numpy',to_numpy()),
])
X_train_cat = cat_pipeline.fit_transform(X_train)
print X_train_cat.shape

print len(data['TotalBsmtSF'].unique())
print len(data['MSSubClass'].unique())
