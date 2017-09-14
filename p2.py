'''kaggle Competition: house price prediction Aug18'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sets import Set
from scipy.stats import skew
import os
os.system('clear')


# read CSV file in Dataframe
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
#data_train.info()

data = data_train.copy()

# seperate X nad y
y_train = data['SalePrice']
X_train = data.drop(['Id','SalePrice'], axis = 1)

X_train.info()

def modify_type(X):
    ''' some features are originally categorical but represented as numerical
    while there are some categorical feats that can be represented as numerical
    '''

    X = X.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40",
    45 : "SC45", 50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75",  80 : "SC80",
    85 : "SC85", 90 : "SC90", 120 : "SC120", 150 : "SC150", 160 : "SC160",
    180 : "SC180", 190 : "SC190"},
    "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
    7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"},
    "ExterQual" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "ExterCond" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "BsmtQual" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "BsmtCond" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "BsmtExposure" : {"Gd": 4, "Av": 3, "Mn": 2, "No" : 1},
    "BsmtFinType1" : {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec" : 3, "LwQ":2,"Unf":1},
    "BsmtFinType2" : {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec" : 3, "LwQ":2,"Unf":1},
    "HeatingQC" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "KitchenQual" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "FireplaceQu" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "GarageQual" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "GarageCond" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "PoolQC" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa" : 2, "Po":1},
    "GarageFinish" : {"Fin": 3, "RFn": 2, "Unf": 1},
    })
    return X
X_train = modify_type(X_train)

def cat_num_list(X):
    ''' seperate numerical and categorical features based on their type'''
    cat = []
    num = []
    for col in  list(X.columns):
        if (X[col].dtype) != ('object'):
            num.append(col)
        else :
            cat.append(col)
    return num, cat

class fill_missing_1(BaseEstimator, TransformerMixin):
    ''' for some categorival attributes None  means it does not have the feature
    this class fill None with DONT '''
    def __init__(self, attr_list = None, fill_by = 0):
        self.attr_list = attr_list
        self.fill_by = fill_by
    def fit(self, X, y = None):
        return self
    def transform(self, X, Y = None):
        if self.attr_list is not None:
            X[self.attr_list] = X[self.attr_list].fillna(self.fill_by)
            return X
        else:
            return X
#
class my_Imputer_cat(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        l = {}
        for col in list(X.columns):
            l[col] = (X[col].value_counts().index[0])
        self.fillingvalue = pd.Series(l)
        return self

    def transform(self, X, y = None):
        return X.fillna(self.fillingvalue)

class my_Imputer_num(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        self.fillingvalue = X.mean()
        return self
    def transform(self, X, y = None):
        return X.fillna(self.fillingvalue)
#
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
#
class to_numpy(BaseEstimator, TransformerMixin):
    def fit(self, X , y = None):
        return self
    def transform(self, X, y = None):
        return X.values

class my_OnehotEncoding(BaseEstimator, TransformerMixin):

    def fit(self, X_cat, y = None):
        return self
    def transform(self, X_cat, y = None):
        X_cat_1hot = pd.get_dummies(X_cat)
        return X_cat_1hot

class feature_list(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        self.attributes_ = X.columns
        return self
    def transform(self, X, y= None):
        return X

class skew_data(BaseEstimator, TransformerMixin):
    def __init__(self, skewness_limit = .75):
        self.skewness_limit = skewness_limit
    def fit(self, X_num, y = None):
        skewness = X_num.apply(lambda x: skew(x))
        large_skewness = skewness[skewness > self.skewness_limit]
        self.num_skew_ = large_skewness.shape[0]
        self.skew_attr_ = large_skewness.index
        return self
    def transform(self, X, y = None):
        X[self.skew_attr_] = np.log1p(X[self.skew_attr_])
        return X

#
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

# use the written function to get a list of numerical and categorical features
num_features, cat_features =  cat_num_list(X_train)
print num_features
print cat_features

# missing value that None means 0(numerical)
att_none_to_0 = [x for x in  Set(att_missing_1).intersection(Set(num_features))]
# meanssing value that None means NO(categirical )
att_none_to_no = [x for x in  Set(att_missing_1).intersection(Set(cat_features))]

# check_pipeline1 = Pipeline([\
# ('num_selector',DataFrameSelector(num_features)),\
# ('missing1', fill_missing_1(attr_list = att_none_to_0, fill_by = 0)),
# ])
# x1 = check_pipeline1.fit_transform(X_train.copy())
# x1.to_csv('cat-data.csv')
# num_features = x1.columns
#
# check_pipeline2 = Pipeline([\
# ('cat_selector', DataFrameSelector(cat_features)),
# ('missing1', fill_missing_1(attr_list = att_none_to_no, fill_by = "NOO")),
# ('encode', my_OnehotEncoding()),
# ])
# x2 = check_pipeline2.fit_transform(X_train)
# x2.to_csv('num_data.csv')
# cat_features = x2.columns



num_pipeline = Pipeline([\
('num_selector',DataFrameSelector(num_features)),\
('missing1', fill_missing_1(attr_list = att_none_to_0, fill_by = 0)),
('get_num_featutes', feature_list()),
('imputer', my_Imputer_num()),
('skew', skew_data()),
('to_numpy',to_numpy()),
('scale', StandardScaler()),
])

#
cat_pipeline = Pipeline([\
('cat_selector', DataFrameSelector(cat_features)),
('missing1', fill_missing_1(attr_list = att_none_to_no, fill_by = "NOO")),
('imputer', my_Imputer_cat()),
('encode', my_OnehotEncoding()),
('get_cat_features', feature_list()),
('to_numpy',to_numpy()),
])
#
# # #
full_pipeline = FeatureUnion(transformer_list = [\
('num_pipeline', num_pipeline),
('cat_pipeline', cat_pipeline),\
])
#
X_train_prepared = full_pipeline.fit_transform(X_train)

#
num_attributes = full_pipeline.transformer_list[0][1].steps[2][1].attributes_
cat_attributes = full_pipeline.transformer_list[1][1].steps[4][1].attributes_
total_attributes = list(num_attributes) + list(cat_attributes)
#
#
y = y_train.values.reshape(-1,1)
y_log = np.log1p(y)
# plot learning curve: for unregulirized regression it is overfitted
# a,b = [],[]
# for N in np.arange(5, x_tr.shape[0], 100):
#     lin_reg = LinearRegression(alpha = 2., random_state = 42)
#     lin_reg.fit(x_tr[:N,:], y_tr[:N])
#     predict_train = lin_reg.predict(x_tr[:N,:])
#     a.append(np.sqrt(mean_squared_error(y_tr[:N] , predict_train)))
#     b.append(np.sqrt(mean_squared_error(y_val , lin_reg.predict(x_val))))
# plt.plot(a,'b')
# plt.plot(b, 'r')

# from sklearn.linear_model import Ridge
# ridge = Ridge(random_state = 42)
# param_grid = {'alpha' : [.03,.1,1,.3,1,3,10]}
# grid = GridSearchCV(estimator = ridge, param_grid = param_grid,
# scoring = 'neg_mean_squared_error', cv = 5, refit = True)
# grid.fit(X_train_prepared , y_log)
# for score, param in zip(grid.cv_results_['mean_test_score'],grid.cv_results_['params']):
#     print (np.sqrt(-score), param)
# print grid.best_params_


# elastic = ElasticNet(random_state = 42,)
# param_grid = {'alpha' : [.05,.1,.4,.8], 'l1_ratio': [0.05, .4,.8,1]}
# grid = GridSearchCV(estimator = elastic, param_grid = param_grid,
# scoring = 'neg_mean_squared_error', cv = 5, refit = True)
# grid.fit(X_train_prepared , y_log)
# for score, param in zip(grid.cv_results_['mean_test_score'],grid.cv_results_['params']):
#     print (np.sqrt(-score), param)
# print grid.best_params_
# l1 = .8 or .4, alpha = .05 0r .1 = 32000


# rnd_reg = RandomForestRegressor(random_state = 42, n_jobs = -1)
# param_grid = {'n_estimators' : [10,100,500]}
# grid = GridSearchCV(estimator = rnd_reg, param_grid = param_grid,
# scoring = 'neg_mean_squared_error', cv = 5, refit = True)
# grid.fit(X_train_prepared , y)
# for score, param in zip(grid.cv_results_['mean_test_score'],grid.cv_results_['params']):
#     print (np.sqrt(-score), param)
# print grid.best_params_
# for 500 & 100  = 30000
##
# rnd_reg = RandomForestRegressor(n_estimators= 100, random_state = 42, n_jobs = -1)
# score_log = cross_val_score(rnd_reg, X_train_prepared, y=y_log,
# scoring='neg_mean_squared_error', cv=5)
# print np.sqrt(-score_log.mean())
# rnd_reg.fit(X_train_prepared, y_log)
# a = zip(rnd_reg.feature_importances_, total_attributes)
# a.sort(key = lambda x :x[0], reverse = True)
# for i in a:
#     print i


grb_reg = GradientBoostingRegressor(random_state = 42)
param_grid = {'learning_rate' : [.05,.1,.5], 'n_estimators' :[10, 100, 500,1000]}
grid = GridSearchCV(estimator = grb_reg, param_grid = param_grid,
scoring = 'neg_mean_squared_error', cv = 5, refit = True)
grid.fit(X_train_prepared , y_log)
for score, param in zip(grid.cv_results_['mean_test_score'],grid.cv_results_['params']):
    print (np.sqrt(-score), param)
print grid.best_params_
##best result = (1000, .05), 25800 near to (500, .1) = 26200

#
# grb_reg = GradientBoostingRegressor(random_state = 42, learning_rate = .1,
# n_estimators = 500)
# score_log = cross_val_score(grb_reg, X_train_prepared, y=y_log,
# scoring='neg_mean_squared_error', cv=10)
# print np.sqrt(-score_log.mean())
