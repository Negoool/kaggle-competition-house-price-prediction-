'''kaggle Competition: house price prediction Aug18'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import scatter_matrix
import os
os.system('clear')


def corrolation_heatmap(data):
    ''' plot corrolation between numerical data in a heatmap style'''
    corr_matrix = data.corr()
    fig, ax = plt.subplots()
    cax = ax.imshow(corr_matrix.values, interpolation= 'nearest', cmap = 'bwr')
    ax.set_yticks(np.arange(len(list(corr_matrix)))+.5)
    ax.set_xticks(np.arange(len(list(corr_matrix)))+.5)
    ax.set_yticklabels(list(corr_matrix), rotation=0,fontsize=8,
    verticalalignment = 'bottom')
    ax.set_xticklabels(list(corr_matrix), rotation=90,fontsize=8,
    horizontalalignment= 'right')
    fig.colorbar(cax)

def partial_corrolation_plot(data, target = 'SalePrice', k = 10):
    ''' plot corrolation between most corrolated numerical attributes'''
    corr_matrix = data.corr()
    # get the most linear corrolated attributes
    important_features = \
    corr_matrix[target].abs().sort_values(ascending = False).index.values[:(k +1)]
    # calculate the corrolation for this subset
    corr_matrix = data[important_features].corr()
    fig, ax = plt.subplots()
    cax = ax.imshow(corr_matrix.values, interpolation= 'nearest', cmap = 'RdPu')
    ax.set_yticks(np.arange(len(list(corr_matrix)))+.5)
    ax.set_xticks(np.arange(len(list(corr_matrix)))+.5)
    ax.set_yticklabels(list(corr_matrix), rotation=0,fontsize=8,
    verticalalignment = 'bottom' )
    ax.set_xticklabels(list(corr_matrix), rotation=90,fontsize=8)
    fig.colorbar(cax)
    # write the value of corrolation on the plot
    for (j,i),label in np.ndenumerate(corr_matrix):
        ax.text(i,j,round(label,3),ha='center',va='center', fontsize =8)


# read CSV file in Dataframe
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# quick look at data
print data.head()
data.info()
#print data.describe()

# visualisation
def data_visualisation(data):
    #make a copy of data for exploration
    explore = data.copy()
    # target value
    explore.hist(column = 'SalePrice',bins = 50)
    plt.figure()
    explore.boxplot(column = 'SalePrice')

    explore.plot.scatter(x = 'GrLivArea', y = 'SalePrice', alpha = .4)
    explore.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')

    # relation of target and some categorical features
    explore.boxplot(column = 'SalePrice', by = 'Neighborhood')
    explore.boxplot(column = 'SalePrice', by = 'YrSold')
    explore.boxplot(column = 'SalePrice', by = 'YearRemodAdd')
    explore.boxplot(column = 'SalePrice', by = 'YearBuilt')
    explore.boxplot(column = 'SalePrice', by = 'Condition1')


    plt.figure()
    explore['MSSubClass'].value_counts().plot.bar()
    plt.title('total number of MSSubClass')

    my_table = pd.crosstab(index = explore['MSSubClass'],
     columns = explore['MSZoning'])
    print type(my_table)
    my_table.plot.bar(stacked = True)


vis = 1
if vis ==1:
    data_visualisation(data)

explore = data.copy()
corrolation_heatmap(explore)
partial_corrolation_plot(explore, target = 'SalePrice', k = 10)


linear_attributes = ["SalePrice","OverallQual", "GrLivArea",
"GarageCars","TotalBsmtSF","FullBath","YearBuilt"]
scatter_matrix(explore[linear_attributes])


print explore['Street'].value_counts()
# print pd.isnull(explore['FireplaceQu']).sum()
plt.figure()
explore['LotFrontage'].plot.hist(bins =50)
plt.show()

# candidates for neew feature

# explore['totbath'] = explore['FullBath'] +  explore['BsmtFullBath'] +explore['HalfBath']
# explore['totrooms'] = explore['FullBath'] + explore['TotRmsAbvGrd']
# explore['bedrooms_per_rooms'] = explore['BedroomAbvGr']/explore['TotRmsAbvGrd']
# corr_matrix = explore.corr()
# print corr_matrix['SalePrice'].sort_values(ascending = False)
# corr_matrix_1 = explore[['GrLivArea', '1stFlrSF']].corr()
# print corr_matrix_1
